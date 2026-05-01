import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from peft import PeftModel
from tqdm import tqdm
import numpy as np
from pathlib import Path
import json
from jiwer import wer

from dataset import RFAudioDataset, collate_fn
from unet import UNet


class RefinementTrainer:

    def __init__(
        self,
        lora_model_path: str,
        refinement_steps: int = 4,
        learning_rate: float = 5e-4,
        weight_decay: float = 1e-3,
        margin_weight: float = 0.5,
        margin_slack: float = 0.0,
        language: str = "en",
    ):
        # -------------------------------------------------------
        # CONFIG: Multi-GPU Split
        # -------------------------------------------------------
        self.device_unet = torch.device("cuda:0")
        self.device_whisper = torch.device("cuda:1")
        print(f"Configuration: UNet on {self.device_unet}, Whisper on {self.device_whisper}")
        # -------------------------------------------------------

        self.refinement_steps = refinement_steps
        self.margin_weight = margin_weight
        self.margin_slack = margin_slack
        self.language = language

        print(f"Loading LoRA-adapted Whisper from {lora_model_path}")

        self.processor = WhisperProcessor.from_pretrained(
            "openai/whisper-large-v3-turbo"
        )

        base_model = WhisperForConditionalGeneration.from_pretrained(
            "openai/whisper-large-v3-turbo",
            torch_dtype=torch.float32  # Keep fp32 for stability
        )

        self.whisper = PeftModel.from_pretrained(base_model, lora_model_path)
        
        # Move Whisper strictly to CUDA:1
        self.whisper.to(self.device_whisper)
        self.whisper.eval()

        for param in self.whisper.parameters():
            param.requires_grad = False

        print("Whisper model frozen and moved to cuda:1")
        
        # Get special token IDs
        self.tokenizer = self.processor.tokenizer
        
        # Extract special tokens
        self.decoder_start_token_id = base_model.config.decoder_start_token_id
        self.lang_token_id = self.tokenizer.convert_tokens_to_ids(f"<|{language}|>")
        self.transcribe_token_id = self.tokenizer.convert_tokens_to_ids("<|transcribe|>")
        self.notimestamps_token_id = self.tokenizer.convert_tokens_to_ids("<|notimestamps|>")
        
        print(f"\nSpecial Token IDs:")
        print(f"  Decoder Start: {self.decoder_start_token_id}")
        print(f"  Language ({language}): {self.lang_token_id}")
        print(f"  Transcribe: {self.transcribe_token_id}")
        print(f"  No Timestamps: {self.notimestamps_token_id}")
        
        # Set generation config
        self.whisper.generation_config.language = language
        self.whisper.generation_config.task = "transcribe"

        # Move UNet strictly to CUDA:0
        self.unet = UNet(
            in_channels=1,
            out_channels=1,
            base_channels=64
        ).to(self.device_unet)

        self.optimizer = torch.optim.AdamW(
            self.unet.parameters(),
            lr=learning_rate,
            betas=(0.9, 0.999),
            weight_decay=weight_decay
        )

        self.scheduler = None
        self.global_step = 0
        self.best_val_wer = float('inf')
        self.patience_counter = 0

    # ------------------------------------------------------------
    # Prepare Batch with Special Tokens
    # ------------------------------------------------------------

    def prepare_labels(self, transcripts, device):
        """
        Prepare labels with proper Whisper special tokens.
        
        Format: <|startoftranscript|><|en|><|transcribe|><|notimestamps|> TRANSCRIPT TEXT <|endoftext|>
        """
        # Tokenize transcripts
        text_tokens = self.tokenizer(
            transcripts,
            padding=False,
            add_special_tokens=False
        ).input_ids
        
        # Build full label sequences
        labels_list = []
        for text_ids in text_tokens:
            full_sequence = [
                self.decoder_start_token_id,  # <|startoftranscript|>
                self.lang_token_id,           # <|en|>
                self.transcribe_token_id,     # <|transcribe|>
                self.notimestamps_token_id,   # <|notimestamps|>
            ] + text_ids + [
                self.tokenizer.eos_token_id   # <|endoftext|>
            ]
            labels_list.append(full_sequence)
        
        # Pad sequences
        max_length = max(len(seq) for seq in labels_list)
        labels_padded = []
        for seq in labels_list:
            padded = seq + [self.tokenizer.pad_token_id] * (max_length - len(seq))
            labels_padded.append(padded)
        
        labels = torch.tensor(labels_padded, dtype=torch.long).to(device)
        
        # Replace padding with -100
        labels = labels.masked_fill(
            labels == self.tokenizer.pad_token_id,
            -100
        )
        
        return labels

    # ------------------------------------------------------------
    # Identity Initialization
    # (Runs entirely on CUDA:0 as it only involves UNet)
    # ------------------------------------------------------------

    def identity_initialization(self, train_loader, max_steps: int = 3000):

        print("\n" + "="*60)
        print("Stage II.A: Identity Initialization")
        print("="*60)

        self.unet.train()
        optimizer = torch.optim.Adam(self.unet.parameters(), lr=1e-3)

        pbar = tqdm(total=max_steps, desc="Identity Init")
        step = 0

        while step < max_steps:
            for batch in train_loader:
                if step >= max_steps:
                    break

                # Input stays on UNet device (cuda:0)
                mel_specs = batch['mel_spectrogram'].to(self.device_unet)
                batch_size = mel_specs.shape[0]
                mel_specs = mel_specs.unsqueeze(1)

                timesteps = torch.randint(
                    1, self.refinement_steps + 1,
                    (batch_size,),
                    device=self.device_unet
                )

                noise_pred = self.unet(mel_specs, timesteps)
                loss = torch.mean(noise_pred ** 2)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                step += 1
                pbar.update(1)
                pbar.set_postfix({'loss': f'{loss.item():.6f}'})

                if loss.item() < 1e-4:
                    pbar.close()
                    return

        pbar.close()

    # ------------------------------------------------------------
    # Refinement Loop with Device Switching
    # ------------------------------------------------------------

    def iterative_refinement(self, mel_spec: torch.Tensor, labels: torch.Tensor):
        # mel_spec is on cuda:0
        # labels is on cuda:1

        batch_size = mel_spec.shape[0]
        refined_specs = [mel_spec] # Keep history on cuda:0
        logits_list = []

        x = mel_spec

        for t in range(1, self.refinement_steps + 1):

            # 1. UNet Step (CUDA:0)
            timesteps = torch.full((batch_size,), t, device=self.device_unet)
            
            noise_pred = self.unet(x, timesteps)
            x = x - noise_pred
            refined_specs.append(x)

            # 2. Whisper Step (CUDA:1)
            # Move tensor to Whisper device. PyTorch autograd handles the gradient flow across devices.
            x_for_whisper = x.squeeze(1).to(self.device_whisper)

            # IMPORTANT: no torch.no_grad() here
            outputs = self.whisper(
                input_features=x_for_whisper,
                labels=labels 
            )

            logits_list.append(outputs.logits)

        return refined_specs, logits_list

    # ------------------------------------------------------------
    # Margin Loss (Computed on CUDA:1)
    # ------------------------------------------------------------

    def compute_margin_loss(self, logits_list: list, labels: torch.Tensor):
        # All inputs here are already on device_whisper

        if len(logits_list) < 2:
            return torch.tensor(0.0, device=self.device_whisper)

        losses = []

        for t in range(1, len(logits_list)):

            prev_logits = logits_list[t-1]
            curr_logits = logits_list[t]

            prev_ce = nn.functional.cross_entropy(
                prev_logits.view(-1, prev_logits.size(-1)),
                labels.view(-1),
                ignore_index=-100
            )

            curr_ce = nn.functional.cross_entropy(
                curr_logits.view(-1, curr_logits.size(-1)),
                labels.view(-1),
                ignore_index=-100
            )

            # Encourage CE to decrease over refinement steps
            margin = torch.relu(curr_ce - prev_ce + self.margin_slack)
            losses.append(margin)

        return torch.stack(losses).mean()

    # ------------------------------------------------------------

    def train_step(self, batch):

        # 1. Load data to UNet device (cuda:0)
        mel_specs = batch['mel_spectrogram'].to(self.device_unet)
        transcripts = batch['transcript']
        mel_specs = mel_specs.unsqueeze(1)

        # 2. Prepare labels directly on Whisper device (cuda:1)
        labels = self.prepare_labels(transcripts, self.device_whisper)

        # 3. Refine
        refined_specs, logits_list = self.iterative_refinement(
            mel_specs,
            labels
        )

        # 4. Final Whisper Pass
        # Get last refined spec from cuda:0 and move to cuda:1
        final_refined = refined_specs[-1].squeeze(1).to(self.device_whisper)

        outputs = self.whisper(
            input_features=final_refined,
            labels=labels
        )

        # 5. Compute Loss (on cuda:1)
        ce_loss = outputs.loss
        margin_loss = self.compute_margin_loss(logits_list, labels)
        total_loss = ce_loss + self.margin_weight * margin_loss

        return total_loss, ce_loss, margin_loss

    # ------------------------------------------------------------

    def train_epoch(self, train_loader, epoch):

        self.unet.train()
        total_loss = 0
        total_ce_loss = 0
        total_margin_loss = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")

        for batch in pbar:

            loss, ce_loss, margin_loss = self.train_step(batch)

            self.optimizer.zero_grad()
            
            # Backward pass will flow from cuda:1 (loss) -> cuda:0 (unet) automatically
            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                self.unet.parameters(),
                max_norm=1.0
            )

            self.optimizer.step()

            if self.scheduler is not None:
                self.scheduler.step()

            total_loss += loss.item()
            total_ce_loss += ce_loss.item()
            total_margin_loss += margin_loss.item()

            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'ce': f'{ce_loss.item():.4f}',
                'margin': f'{margin_loss.item():.4f}'
            })

        num_batches = len(train_loader)

        return {
            'loss': total_loss / num_batches,
            'ce_loss': total_ce_loss / num_batches,
            'margin_loss': total_margin_loss / num_batches
        }

    # ------------------------------------------------------------

    @torch.no_grad()
    def evaluate(self, val_loader):

        self.unet.eval()
        total_loss = 0
        all_predictions = []
        all_references = []

        for batch in tqdm(val_loader, desc="Evaluating"):

            # Data to cuda:0
            mel_specs = batch['mel_spectrogram'].to(self.device_unet)
            transcripts = batch['transcript']
            mel_specs = mel_specs.unsqueeze(1)

            x = mel_specs

            # Refinement loop on cuda:0
            for t in range(1, self.refinement_steps + 1):
                timesteps = torch.full(
                    (x.shape[0],), t,
                    device=self.device_unet
                )
                noise_pred = self.unet(x, timesteps)
                x = x - noise_pred

            # Move result to cuda:1 for Whisper
            final_refined = x.squeeze(1).to(self.device_whisper)

            # Labels to cuda:1
            labels = self.prepare_labels(transcripts, self.device_whisper)

            outputs = self.whisper(
                input_features=final_refined,
                labels=labels
            )

            total_loss += outputs.loss.item()

            # Generate on cuda:1 with proper language and task settings
            generated_ids = self.whisper.generate(
                final_refined,
                max_length=80,
                num_beams=3,
                early_stopping=True,
                repetition_penalty=1.2,          
                no_repeat_ngram_size=3,
                task="transcribe",
                eos_token_id=self.processor.tokenizer.eos_token_id,  # ADD THIS
            )

            predictions = self.processor.batch_decode(
                generated_ids,
                skip_special_tokens=True
            )

            all_predictions.extend(predictions)
            all_references.extend(transcripts)
            
            # Print first few examples
            if len(all_predictions) <= 5:
                print(f"\nPrediction: {all_predictions[-1]}")
                print(f"Reference:  {all_references[-1]}")

        avg_loss = total_loss / len(val_loader)
        wer_score = wer(all_references, all_predictions) * 100

        return avg_loss, wer_score

    # ------------------------------------------------------------

    def save_checkpoint(self, path, val_wer):

        os.makedirs(path, exist_ok=True)

        torch.save({
            'unet_state_dict': self.unet.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_wer': val_wer,
            'language': self.language,
            'special_tokens': {
                'decoder_start_token_id': self.decoder_start_token_id,
                'lang_token_id': self.lang_token_id,
                'transcribe_token_id': self.transcribe_token_id,
                'notimestamps_token_id': self.notimestamps_token_id
            }
        }, os.path.join(path, 'unet_checkpoint.pt'))

        print(f"Checkpoint saved to {path}")

    # ------------------------------------------------------------

    def train(
        self,
        train_loader,
        val_loader,
        num_epochs: int = 50,
        patience: int = 400,
        output_dir: str = "outputs/refinement"
    ):

        self.identity_initialization(train_loader)

        total_steps = len(train_loader) * num_epochs
        warmup_steps = 500

        def lr_lambda(step):
            if step < warmup_steps:
                return step / warmup_steps
            else:
                progress = (step - warmup_steps) / (total_steps - warmup_steps)
                return 0.5 * (1 + np.cos(np.pi * progress))

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lr_lambda
        )

        for epoch in range(1, num_epochs + 1):

            train_metrics = self.train_epoch(train_loader, epoch)
            val_loss, val_wer = self.evaluate(val_loader)

            print(f"\nEpoch {epoch}/{num_epochs}")
            print(f"  Train Loss: {train_metrics['loss']:.4f}")
            print(f"  Val Loss: {val_loss:.4f}")
            print(f"  Val WER: {val_wer:.2f}%")

            if val_wer < self.best_val_wer:
                self.best_val_wer = val_wer
                self.patience_counter = 0

                checkpoint_path = os.path.join(output_dir, "best_model")
                self.save_checkpoint(checkpoint_path, val_wer)

                print(f"  ✓ New best model! (WER: {val_wer:.2f}%)")
            else:
                self.patience_counter += 1

            if self.patience_counter >= patience:
                print(f"\nEarly stopping at epoch {epoch}")
                break

        print(f"\nTraining completed!")
        print(f"Best validation WER: {self.best_val_wer:.2f}%")


def main():
    parser = argparse.ArgumentParser(description="Stage II: Acoustic Refinement")
    
    # Data arguments
    parser.add_argument('--audio_dir', type=str, required=True)
    parser.add_argument('--transcript_dir', type=str, required=True)
    parser.add_argument('--val_split', type=float, default=0.1)
    
    # Model arguments
    parser.add_argument('--lora_model', type=str, required=True,
                       help='Path to trained LoRA model')
    parser.add_argument('--refinement_steps', type=int, default=4)
    parser.add_argument('--language', type=str, default='en',
                       help='Language code (e.g., en, es, fr)')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--learning_rate', type=float, default=5e-4)
    parser.add_argument('--margin_weight', type=float, default=0.5)
    parser.add_argument('--patience', type=int, default=5)
    
    # Output arguments
    parser.add_argument('--output_dir', type=str, default='outputs/refinement')
    parser.add_argument('--seed', type=int, default=42)
    
    args = parser.parse_args()
    
    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Load dataset
    print("Loading dataset...")
    full_dataset = RFAudioDataset(
        audio_dir=args.audio_dir,
        transcript_dir=args.transcript_dir
    )
    
    # Split
    val_size = int(len(full_dataset) * args.val_split)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(args.seed)
    )
    
    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    
    # Create loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=4
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=4
    )
    
    # Initialize trainer
    trainer = RefinementTrainer(
        lora_model_path=args.lora_model,
        refinement_steps=args.refinement_steps,
        learning_rate=args.learning_rate,
        margin_weight=args.margin_weight,
        language=args.language
    )
    
    # Train
    trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=args.epochs,
        patience=args.patience,
        output_dir=args.output_dir
    )


if __name__ == "__main__":
    main()