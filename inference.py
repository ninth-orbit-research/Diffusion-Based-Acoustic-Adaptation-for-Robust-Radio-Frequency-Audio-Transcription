"""
Inference Script for RF-Domain ASR

Applies the complete two-stage pipeline:
1. Load audio and convert to mel spectrogram
2. Apply iterative U-Net refinement (T=4 steps)
3. Transcribe using LoRA-adapted Whisper
"""

import argparse
import torch
import librosa
import soundfile as sf
import numpy as np
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from peft import PeftModel
from pathlib import Path

from unet import UNet


class RFASRInference:
    """
    Complete inference pipeline for RF-domain ASR.
    """
    
    def __init__(
        self,
        lora_model_path: str,
        refinement_model_path: str,
        refinement_steps: int = 4,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.device = device
        self.refinement_steps = refinement_steps
        
        print(f"Initializing RF-ASR pipeline on {device}")
        print(f"LoRA model: {lora_model_path}")
        print(f"Refinement model: {refinement_model_path}")
        
        # Load processor
        self.processor = WhisperProcessor.from_pretrained("openai/whisper-large-v3-turbo")
        
        # Load LoRA-adapted Whisper
        print("\nLoading LoRA-adapted Whisper...")
        base_model = WhisperForConditionalGeneration.from_pretrained(
            "openai/whisper-large-v3-turbo",
            torch_dtype=torch.float16 if device == "cuda" else torch.float32
        )
        self.whisper = PeftModel.from_pretrained(base_model, lora_model_path)
        self.whisper.to(device)
        self.whisper.eval()
        
        # Load refinement network
        print("Loading refinement network...")
        self.unet = UNet().to(device)
        
        checkpoint = torch.load(
            refinement_model_path,
            map_location=device
        )
        self.unet.load_state_dict(checkpoint['unet_state_dict'])
        self.unet.eval()
        
        print("\nPipeline ready for inference!")
    
    def load_audio(self, audio_path: str) -> np.ndarray:
        """
        Load and preprocess audio file.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Audio waveform at 16kHz
        """
        # Load audio
        audio, sr = sf.read(audio_path)
        
        # Convert to mono
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        
        # Resample to 16kHz
        if sr != 16000:
            audio = librosa.resample(
                audio,
                orig_sr=sr,
                target_sr=16000,
                res_type='kaiser_best'
            )
        
        return audio
    
    def audio_to_mel_spectrogram(self, audio: np.ndarray) -> np.ndarray:
        """
        Convert audio to log-mel spectrogram.
        
        Args:
            audio: Audio waveform at 16kHz
            
        Returns:
            Log-mel spectrogram (80, T)
        """
        # STFT parameters
        n_fft = 400
        hop_length = 160
        win_length = 400
        
        # Compute STFT
        stft = librosa.stft(
            audio,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window='hann'
        )
        
        # Power spectrogram
        power_spec = np.abs(stft) ** 2
        
        # Mel filterbank
        mel_basis = librosa.filters.mel(
            sr=16000,
            n_fft=n_fft,
            n_mels=80,
            fmin=0,
            fmax=8000
        )
        
        # Apply filterbank
        mel_spec = mel_basis @ power_spec
        
        # Log compression
        log_mel_spec = np.log(mel_spec + 1e-10)
        
        return log_mel_spec
    
    @torch.no_grad()
    def refine_spectrogram(self, mel_spec: torch.Tensor) -> torch.Tensor:
        """
        Apply iterative refinement to spectrogram.
        
        Args:
            mel_spec: Input mel spectrogram (1, 1, 80, T)
            
        Returns:
            Refined mel spectrogram (1, 1, 80, T)
        """
        x = mel_spec
        
        print(f"\nApplying {self.refinement_steps}-step refinement...")
        
        for t in range(1, self.refinement_steps + 1):
            # Create timestep
            timesteps = torch.tensor([t], device=self.device)
            
            # Predict noise
            noise_pred = self.unet(x, timesteps)
            
            # Subtract noise
            x = x - noise_pred
            
            print(f"  Step {t}/{self.refinement_steps} complete")
        
        return x
    
    @torch.no_grad()
    def transcribe(
        self,
        audio_path: str,
        apply_refinement: bool = True,
        return_intermediate: bool = False
    ) -> dict:
        """
        Transcribe RF-degraded audio.
        
        Args:
            audio_path: Path to audio file
            apply_refinement: Whether to apply acoustic refinement
            return_intermediate: Return intermediate results
            
        Returns:
            Dictionary with transcription and optional intermediate results
        """
        print(f"\nTranscribing: {audio_path}")
        
        # Load audio
        audio = self.load_audio(audio_path)
        print(f"Audio duration: {len(audio) / 16000:.2f}s")
        
        # Convert to mel spectrogram
        mel_spec = self.audio_to_mel_spectrogram(audio)
        
        # Convert to tensor
        mel_tensor = torch.from_numpy(mel_spec).float().to(self.device)
        mel_tensor = mel_tensor.unsqueeze(0).unsqueeze(0)  # (1, 1, 80, T)
        
        results = {'audio_path': audio_path}
        
        # Optional: Transcribe without refinement (baseline)
        if return_intermediate:
            print("\nGenerating baseline transcription (no refinement)...")
            baseline_spec = mel_tensor.squeeze(1)  # (1, 80, T)
            baseline_ids = self.whisper.generate(
                baseline_spec,
                max_length=200,
                num_beams=5
            )
            baseline_text = self.processor.batch_decode(
                baseline_ids,
                skip_special_tokens=True
            )[0]
            results['baseline_transcript'] = baseline_text
            print(f"Baseline: {baseline_text}")
        
        # Apply refinement if requested
        if apply_refinement:
            refined_spec = self.refine_spectrogram(mel_tensor)
        else:
            refined_spec = mel_tensor
        
        # Transcribe
        print("\nGenerating final transcription...")
        final_spec = refined_spec.squeeze(1)  # (1, 80, T)
        
        generated_ids = self.whisper.generate(
            final_spec,
            max_length=200,
            num_beams=5,
            early_stopping=True
        )
        
        final_text = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True
        )[0]
        
        results['transcript'] = final_text
        results['refinement_applied'] = apply_refinement
        
        print(f"\nFinal: {final_text}")
        
        return results
    
    def transcribe_batch(
        self,
        audio_paths: list,
        apply_refinement: bool = True
    ) -> list:
        """
        Transcribe multiple audio files.
        
        Args:
            audio_paths: List of audio file paths
            apply_refinement: Whether to apply refinement
            
        Returns:
            List of transcription results
        """
        results = []
        
        for audio_path in audio_paths:
            try:
                result = self.transcribe(audio_path, apply_refinement)
                results.append(result)
            except Exception as e:
                print(f"Error processing {audio_path}: {e}")
                results.append({
                    'audio_path': audio_path,
                    'transcript': None,
                    'error': str(e)
                })
        
        return results


def main():
    parser = argparse.ArgumentParser(description="RF-ASR Inference")
    
    # Model arguments
    parser.add_argument('--lora_model', type=str, required=True,
                       help='Path to LoRA model directory')
    parser.add_argument('--refinement_model', type=str, required=True,
                       help='Path to refinement model checkpoint (.pt)')
    parser.add_argument('--refinement_steps', type=int, default=4,
                       help='Number of refinement steps')
    
    # Input arguments
    parser.add_argument('--audio_file', type=str,
                       help='Single audio file to transcribe')
    parser.add_argument('--audio_dir', type=str,
                       help='Directory of audio files to transcribe')
    
    # Output arguments
    parser.add_argument('--output_file', type=str,
                       help='Output file for transcript')
    parser.add_argument('--no_refinement', action='store_true',
                       help='Disable acoustic refinement (baseline)')
    parser.add_argument('--show_baseline', action='store_true',
                       help='Show baseline transcription without refinement')
    
    args = parser.parse_args()
    
    if not args.audio_file and not args.audio_dir:
        parser.error("Either --audio_file or --audio_dir must be specified")
    
    # Initialize pipeline
    pipeline = RFASRInference(
        lora_model_path=args.lora_model,
        refinement_model_path=args.refinement_model,
        refinement_steps=args.refinement_steps
    )
    
    # Process single file
    if args.audio_file:
        results = pipeline.transcribe(
            args.audio_file,
            apply_refinement=not args.no_refinement,
            return_intermediate=args.show_baseline
        )
        
        # Save output
        if args.output_file:
            with open(args.output_file, 'w') as f:
                f.write(results['transcript'])
            print(f"\nTranscript saved to {args.output_file}")
        
        # Print results
        print("\n" + "="*60)
        print("RESULTS")
        print("="*60)
        
        if args.show_baseline and 'baseline_transcript' in results:
            print(f"\nBaseline (no refinement):")
            print(f"  {results['baseline_transcript']}")
        
        print(f"\nFinal transcription:")
        print(f"  {results['transcript']}")
        print(f"\nRefinement applied: {results['refinement_applied']}")
    
    # Process directory
    elif args.audio_dir:
        audio_dir = Path(args.audio_dir)
        audio_files = sorted(audio_dir.glob("*.wav"))
        
        print(f"\nFound {len(audio_files)} audio files")
        
        results = pipeline.transcribe_batch(
            [str(f) for f in audio_files],
            apply_refinement=not args.no_refinement
        )
        
        # Save results
        if args.output_file:
            with open(args.output_file, 'w') as f:
                for result in results:
                    if result['transcript']:
                        f.write(f"{Path(result['audio_path']).stem}: {result['transcript']}\n")
            print(f"\nTranscripts saved to {args.output_file}")
        
        # Print summary
        successful = sum(1 for r in results if r['transcript'])
        print(f"\nProcessed {successful}/{len(results)} files successfully")


if __name__ == "__main__":
    main()
