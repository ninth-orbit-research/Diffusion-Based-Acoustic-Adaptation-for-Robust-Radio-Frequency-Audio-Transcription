"""
Stage I: Parameter-Efficient Domain Adaptation using LoRA

FIXED VERSION - Based on working Whisper fine-tuning patterns
Key changes:
- Proper forced_decoder_ids configuration
- No manual special token insertion in labels
- Correct generation settings
- Fixed suppress_tokens and use_cache
"""

import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from peft import LoraConfig, get_peft_model
from tqdm import tqdm
import numpy as np
from pathlib import Path
import json
from jiwer import wer

from dataset import RFAudioDataset, collate_fn


class LoRATrainer:
    """
    Trainer for LoRA adaptation of Whisper on RF-degraded audio.
    """
    
    def __init__(
        self,
        model_name: str = "openai/whisper-large-v3-turbo",
        lora_rank: int = 64,
        lora_alpha: int = 32,
        learning_rate: float = 3e-4,
        weight_decay: float = 1e-2,
        warmup_steps: int = 100,
        language: str = "en",
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.device = device
        self.learning_rate = learning_rate
        self.warmup_steps = warmup_steps
        self.language = language
        
        print(f"Loading Whisper model: {model_name}")
        
        # Load processor and model
        self.processor = WhisperProcessor.from_pretrained(model_name)
        base_model = WhisperForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.float32
        )
        
        # CRITICAL: Configure model for training
        base_model.config.use_cache = False
        
        # Set generation config (NOT base config)
        base_model.generation_config.language = language
        base_model.generation_config.task = "transcribe"
        base_model.generation_config.forced_decoder_ids = None
        base_model.generation_config.suppress_tokens = []
        
        print(f"\nModel Configuration:")
        print(f"  Language: {language}")
        print(f"  Task: transcribe")
        print(f"  config.use_cache: False")
        print(f"  generation_config.forced_decoder_ids: None")
        print(f"  generation_config.suppress_tokens: []")
        
        # Configure LoRA
        lora_config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=0.1,  # Add dropout for regularization
            target_modules=[
                "q_proj",
                "k_proj", 
                "v_proj",
                "out_proj"
            ],
            bias="none",
            inference_mode=False
        )
        
        # Apply LoRA to model
        self.model = get_peft_model(base_model, lora_config)
        self.model.to(device)
        
        # Print trainable parameters
        self.model.print_trainable_parameters()
        
        # Setup optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            betas=(0.9, 0.999),
            weight_decay=weight_decay
        )
        
        # Initialize learning rate scheduler (set in train method)
        self.scheduler = None
        
        # Training state
        self.global_step = 0
        self.best_val_wer = float('inf')
        self.patience_counter = 0
        self.checkpoint_scores = []
    
    def prepare_batch(self, batch):
        """
        Prepare batch for training.
        
        KEY CHANGE: Let Whisper handle special tokens automatically
        through forced_decoder_ids, don't manually insert them in labels.
        """
        mel_specs = batch['mel_spectrogram'].to(self.device)
        transcripts = batch['transcript']
        
        # Tokenize transcripts normally - NO manual special token insertion
        labels = self.processor.tokenizer(
            transcripts,
            padding=True,
            truncation=True,
            max_length=448,
            return_tensors="pt"
        ).input_ids.to(self.device)
        
        # Replace padding token id with -100 so it's ignored in loss
        labels = labels.masked_fill(
            labels == self.processor.tokenizer.pad_token_id,
            -100
        )
        
        return {
            'input_features': mel_specs,
            'labels': labels
        }
    
    def train_epoch(self, train_loader, epoch):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        num_batches = len(train_loader)
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
        for batch_idx, batch in enumerate(pbar):
            # Prepare inputs
            inputs = self.prepare_batch(batch)
            
            # Forward pass
            outputs = self.model(**inputs)
            loss = outputs.loss
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            # Optimizer step
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()
            
            # Update metrics
            total_loss += loss.item()
            self.global_step += 1
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'lr': f'{self.scheduler.get_last_lr()[0]:.2e}'
            })
        
        avg_loss = total_loss / num_batches
        return avg_loss
    
    @torch.no_grad()
    def evaluate(self, val_loader):
        """Evaluate on validation set."""
        self.model.eval()
        total_loss = 0
        all_predictions = []
        all_references = []
        
        for batch in tqdm(val_loader, desc="Evaluating"):
            # Prepare inputs
            inputs = self.prepare_batch(batch)
            
            # Compute loss
            outputs = self.model(**inputs)
            total_loss += outputs.loss.item()
            
            # Generate predictions
            generated_ids = self.model.generate(
                inputs['input_features'],
                max_length=80,
                num_beams=3,
                early_stopping=True,
                repetition_penalty=1.2,          
                no_repeat_ngram_size=3,
                task="transcribe",
                eos_token_id=self.processor.tokenizer.eos_token_id,  # ADD THIS
            )
            
            # Decode predictions
            predictions = self.processor.batch_decode(
                generated_ids,
                skip_special_tokens=True
            )
            
            all_predictions.extend(predictions)
            all_references.extend(batch['transcript'])
            
            # Print first few examples for debugging
            if len(all_predictions) <= 3:
                print(f"\n{'='*60}")
                print(f"Prediction: {all_predictions[-1]}")
                print(f"Reference:  {all_references[-1]}")
                print(f"{'='*60}")
        
        avg_loss = total_loss / len(val_loader)
        
        # Compute WER
        wer_score = wer(all_references, all_predictions)
        
        return avg_loss, wer_score * 100
    
    def save_checkpoint(self, path, val_wer):
        """Save model checkpoint."""
        os.makedirs(path, exist_ok=True)
        
        # Save LoRA adapter
        self.model.save_pretrained(path)
        
        # Save processor
        self.processor.save_pretrained(path)
        
        # Save training state
        state = {
            'global_step': self.global_step,
            'best_val_wer': self.best_val_wer,
            'val_wer': val_wer,
            'language': self.language
        }
        
        with open(os.path.join(path, 'training_state.json'), 'w') as f:
            json.dump(state, f, indent=2)
        
        print(f"Checkpoint saved to {path}")
    
    def train(
        self,
        train_loader,
        val_loader,
        num_epochs: int = 12,
        patience: int = 4,
        output_dir: str = "outputs/lora"
    ):
        """
        Main training loop with early stopping.
        """
        # Setup learning rate scheduler
        total_steps = len(train_loader) * num_epochs
        
        def lr_lambda(step):
            if step < self.warmup_steps:
                return step / self.warmup_steps
            else:
                progress = (step - self.warmup_steps) / (total_steps - self.warmup_steps)
                return 0.5 * (1 + np.cos(np.pi * progress))
        
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lr_lambda
        )
        
        print(f"\nStarting training for {num_epochs} epochs")
        print(f"Total training steps: {total_steps}")
        print(f"Warmup steps: {self.warmup_steps}")
        print(f"Early stopping patience: {patience}")
        
        for epoch in range(1, num_epochs + 1):
            # Train
            train_loss = self.train_epoch(train_loader, epoch)
            
            # Evaluate
            val_loss, val_wer = self.evaluate(val_loader)
            
            print(f"\nEpoch {epoch}/{num_epochs}")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val Loss: {val_loss:.4f}")
            print(f"  Val WER: {val_wer:.2f}%")
            
            # Check if best model
            if val_wer < self.best_val_wer:
                self.best_val_wer = val_wer
                self.patience_counter = 0
                
                # Save best checkpoint
                checkpoint_path = os.path.join(output_dir, "best_model")
                self.save_checkpoint(checkpoint_path, val_wer)
                
                print(f"  ✓ New best model! (WER: {val_wer:.2f}%)")
            else:
                self.patience_counter += 1
                print(f"  No improvement ({self.patience_counter}/{patience})")
            
            # Save periodic checkpoint
            if epoch % 5 == 0:
                checkpoint_path = os.path.join(output_dir, f"checkpoint_epoch_{epoch}")
                self.save_checkpoint(checkpoint_path, val_wer)
            
            # Early stopping
            if self.patience_counter >= patience:
                print(f"\nEarly stopping triggered after {epoch} epochs")
                print(f"Best validation WER: {self.best_val_wer:.2f}%")
                break
        
        print(f"\nTraining completed!")
        print(f"Best validation WER: {self.best_val_wer:.2f}%")


def main():
    parser = argparse.ArgumentParser(description="Stage I: LoRA Adaptation of Whisper")
    
    # Data arguments
    parser.add_argument('--audio_dir', type=str, required=True,
                       help='Directory containing RF audio files')
    parser.add_argument('--transcript_dir', type=str, required=True,
                       help='Directory containing transcript files')
    parser.add_argument('--val_split', type=float, default=0.1,
                       help='Validation set split ratio')
    
    # Model arguments
    parser.add_argument('--model_name', type=str, default='openai/whisper-large-v3-turbo',
                       help='Whisper model to use')
    parser.add_argument('--lora_rank', type=int, default=64,
                       help='LoRA rank (r)')
    parser.add_argument('--lora_alpha', type=int, default=32,
                       help='LoRA scaling factor (alpha)')
    parser.add_argument('--language', type=str, default='en',
                       help='Language code (e.g., en, es, fr)')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=12,
                       help='Maximum number of epochs')
    parser.add_argument('--batch_size', type=int, default=4,
                       help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=3e-4,
                       help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-2,
                       help='Weight decay for AdamW')
    parser.add_argument('--warmup_steps', type=int, default=100,
                       help='Number of warmup steps')
    parser.add_argument('--patience', type=int, default=4,
                       help='Early stopping patience')
    
    # Output arguments
    parser.add_argument('--output_dir', type=str, default='outputs/lora',
                       help='Directory to save model checkpoints')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    args = parser.parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Load dataset
    print("Loading dataset...")
    full_dataset = RFAudioDataset(
        audio_dir=args.audio_dir,
        transcript_dir=args.transcript_dir,
        augment=False
    )
    
    # Split into train/val
    val_size = int(len(full_dataset) * args.val_split)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(args.seed)
    )
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True
    )
    
    # Initialize trainer
    trainer = LoRATrainer(
        model_name=args.model_name,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
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