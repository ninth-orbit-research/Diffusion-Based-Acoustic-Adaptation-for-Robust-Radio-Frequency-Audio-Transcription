"""
Evaluation Script for RF-Domain ASR

Compares performance across different configurations:
1. Zero-shot Whisper (no adaptation)
2. LoRA-adapted Whisper (Stage I only)
3. Full pipeline (Stage I + Stage II)

Computes WER across different SNR conditions.
"""

import argparse
import torch
import numpy as np
from torch.utils.data import DataLoader
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from peft import PeftModel
from tqdm import tqdm
from jiwer import wer, cer
import json
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

from dataset import RFAudioDataset, collate_fn
from unet import UNet


class RFASREvaluator:
    """
    Comprehensive evaluator for RF-ASR system.
    """
    
    def __init__(
        self,
        lora_model_path: str = None,
        refinement_model_path: str = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.device = device
        
        # Load processor
        self.processor = WhisperProcessor.from_pretrained("openai/whisper-large-v3-turbo")
        
        # Load zero-shot model
        print("Loading zero-shot Whisper...")
        self.zero_shot_model = WhisperForConditionalGeneration.from_pretrained(
            "openai/whisper-large-v3-turbo",
            torch_dtype=torch.float16 if device == "cuda" else torch.float32
        ).to(device).eval()
        
        # Load LoRA model if provided
        self.lora_model = None
        if lora_model_path:
            print(f"Loading LoRA model from {lora_model_path}")
            base = WhisperForConditionalGeneration.from_pretrained(
                "openai/whisper-large-v3-turbo",
                torch_dtype=torch.float16 if device == "cuda" else torch.float32
            )
            self.lora_model = PeftModel.from_pretrained(base, lora_model_path)
            self.lora_model.to(device).eval()
        
        # Load refinement model if provided
        self.unet = None
        if refinement_model_path:
            print(f"Loading refinement model from {refinement_model_path}")
            self.unet = UNet().to(device)
            checkpoint = torch.load(refinement_model_path, map_location=device)
            self.unet.load_state_dict(checkpoint['unet_state_dict'])
            self.unet.eval()
    
    @torch.no_grad()
    def evaluate_model(
        self,
        model,
        dataloader,
        apply_refinement: bool = False,
        refinement_steps: int = 4
    ):
        """
        Evaluate a model on a dataset.
        
        Args:
            model: Whisper model to evaluate
            dataloader: DataLoader with test data
            apply_refinement: Whether to apply U-Net refinement
            refinement_steps: Number of refinement steps
            
        Returns:
            Dictionary with metrics
        """
        all_predictions = []
        all_references = []
        
        for batch in tqdm(dataloader, desc="Evaluating"):
            mel_specs = batch['mel_spectrogram'].to(self.device)
            transcripts = batch['transcript']
            
            # Apply refinement if enabled
            if apply_refinement and self.unet is not None:
                # Add channel dimension
                mel_specs = mel_specs.unsqueeze(1)
                
                # Iterative refinement
                x = mel_specs
                for t in range(1, refinement_steps + 1):
                    batch_size = x.shape[0]
                    timesteps = torch.full((batch_size,), t, device=self.device)
                    noise_pred = self.unet(x, timesteps)
                    x = x - noise_pred
                
                # Remove channel dimension
                mel_specs = x.squeeze(1)
            
            # Generate predictions
            generated_ids = model.generate(
                mel_specs,
                max_length=200,
                num_beams=5,
                early_stopping=True
            )
            
            # Decode
            predictions = self.processor.batch_decode(
                generated_ids,
                skip_special_tokens=True
            )
            
            all_predictions.extend(predictions)
            all_references.extend(transcripts)
        
        # Compute metrics
        wer_score = wer(all_references, all_predictions) * 100
        cer_score = cer(all_references, all_predictions) * 100
        
        return {
            'wer': wer_score,
            'cer': cer_score,
            'predictions': all_predictions,
            'references': all_references
        }
    
    def comprehensive_evaluation(self, test_loader):
        """
        Run comprehensive evaluation across all configurations.
        """
        results = {}
        
        # 1. Zero-shot baseline
        print("\n" + "="*60)
        print("Evaluating Zero-Shot Whisper")
        print("="*60)
        results['zero_shot'] = self.evaluate_model(
            self.zero_shot_model,
            test_loader,
            apply_refinement=False
        )
        print(f"WER: {results['zero_shot']['wer']:.2f}%")
        print(f"CER: {results['zero_shot']['cer']:.2f}%")
        
        # 2. LoRA-adapted (Stage I only)
        if self.lora_model is not None:
            print("\n" + "="*60)
            print("Evaluating LoRA-Adapted Whisper (Stage I)")
            print("="*60)
            results['lora_only'] = self.evaluate_model(
                self.lora_model,
                test_loader,
                apply_refinement=False
            )
            print(f"WER: {results['lora_only']['wer']:.2f}%")
            print(f"CER: {results['lora_only']['cer']:.2f}%")
            
            # Relative improvement
            improvement = (
                (results['zero_shot']['wer'] - results['lora_only']['wer']) /
                results['zero_shot']['wer'] * 100
            )
            print(f"Relative improvement: {improvement:.1f}%")
        
        # 3. Full pipeline (Stage I + Stage II)
        if self.lora_model is not None and self.unet is not None:
            print("\n" + "="*60)
            print("Evaluating Full Pipeline (Stage I + Stage II)")
            print("="*60)
            results['full_pipeline'] = self.evaluate_model(
                self.lora_model,
                test_loader,
                apply_refinement=True,
                refinement_steps=4
            )
            print(f"WER: {results['full_pipeline']['wer']:.2f}%")
            print(f"CER: {results['full_pipeline']['cer']:.2f}%")
            
            # Relative improvements
            improvement_vs_zero_shot = (
                (results['zero_shot']['wer'] - results['full_pipeline']['wer']) /
                results['zero_shot']['wer'] * 100
            )
            improvement_vs_lora = (
                (results['lora_only']['wer'] - results['full_pipeline']['wer']) /
                results['lora_only']['wer'] * 100
            )
            
            print(f"Relative improvement vs zero-shot: {improvement_vs_zero_shot:.1f}%")
            print(f"Relative improvement vs LoRA: {improvement_vs_lora:.1f}%")
        
        return results
    
    def ablation_study(self, test_loader):
        """
        Ablation study: varying number of refinement steps.
        """
        if self.lora_model is None or self.unet is None:
            print("Ablation study requires both LoRA and refinement models")
            return None
        
        print("\n" + "="*60)
        print("Ablation Study: Refinement Steps")
        print("="*60)
        
        results = {}
        
        for steps in [1, 2, 4, 8]:
            print(f"\nTesting with {steps} refinement steps...")
            
            metrics = self.evaluate_model(
                self.lora_model,
                test_loader,
                apply_refinement=True,
                refinement_steps=steps
            )
            
            results[f'steps_{steps}'] = metrics
            print(f"  WER: {metrics['wer']:.2f}%")
        
        return results
    
    def save_results(self, results: dict, output_path: str):
        """Save evaluation results to JSON."""
        # Convert results to serializable format
        serializable_results = {}
        
        for config_name, metrics in results.items():
            serializable_results[config_name] = {
                'wer': float(metrics['wer']),
                'cer': float(metrics['cer'])
            }
        
        with open(output_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        print(f"\nResults saved to {output_path}")
    
    def plot_results(self, results: dict, output_path: str):
        """Create visualization of results."""
        # Extract WER scores
        configs = []
        wer_scores = []
        
        for config_name, metrics in results.items():
            configs.append(config_name.replace('_', ' ').title())
            wer_scores.append(metrics['wer'])
        
        # Create bar plot
        plt.figure(figsize=(10, 6))
        sns.set_style("whitegrid")
        
        colors = sns.color_palette("husl", len(configs))
        bars = plt.bar(configs, wer_scores, color=colors, alpha=0.8)
        
        # Add value labels on bars
        for bar, score in zip(bars, wer_scores):
            height = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width()/2.,
                height,
                f'{score:.1f}%',
                ha='center',
                va='bottom',
                fontsize=12,
                fontweight='bold'
            )
        
        plt.xlabel('Configuration', fontsize=14, fontweight='bold')
        plt.ylabel('Word Error Rate (%)', fontsize=14, fontweight='bold')
        plt.title('RF-Domain ASR Performance Comparison', fontsize=16, fontweight='bold')
        plt.xticks(rotation=15, ha='right')
        plt.tight_layout()
        
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {output_path}")
        plt.close()


def main():
    parser = argparse.ArgumentParser(description="RF-ASR Evaluation")
    
    # Data arguments
    parser.add_argument('--audio_dir', type=str, required=True)
    parser.add_argument('--transcript_dir', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=8)
    
    # Model arguments
    parser.add_argument('--lora_model', type=str,
                       help='Path to LoRA model')
    parser.add_argument('--refinement_model', type=str,
                       help='Path to refinement model checkpoint')
    
    # Evaluation options
    parser.add_argument('--ablation', action='store_true',
                       help='Run ablation study on refinement steps')
    
    # Output arguments
    parser.add_argument('--output_dir', type=str, default='outputs/evaluation')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load test dataset
    print("Loading test dataset...")
    test_dataset = RFAudioDataset(
        audio_dir=args.audio_dir,
        transcript_dir=args.transcript_dir
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=4
    )
    
    print(f"Test samples: {len(test_dataset)}")
    
    # Initialize evaluator
    evaluator = RFASREvaluator(
        lora_model_path=args.lora_model,
        refinement_model_path=args.refinement_model
    )
    
    # Run comprehensive evaluation
    results = evaluator.comprehensive_evaluation(test_loader)
    
    # Save results
    evaluator.save_results(
        results,
        output_dir / 'evaluation_results.json'
    )
    
    # Create visualization
    evaluator.plot_results(
        results,
        output_dir / 'performance_comparison.png'
    )
    
    # Optional: ablation study
    if args.ablation:
        ablation_results = evaluator.ablation_study(test_loader)
        if ablation_results:
            evaluator.save_results(
                ablation_results,
                output_dir / 'ablation_results.json'
            )
            evaluator.plot_results(
                ablation_results,
                output_dir / 'ablation_comparison.png'
            )
    
    # Print summary
    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)
    
    for config_name, metrics in results.items():
        print(f"\n{config_name.replace('_', ' ').title()}:")
        print(f"  WER: {metrics['wer']:.2f}%")
        print(f"  CER: {metrics['cer']:.2f}%")


if __name__ == "__main__":
    main()
