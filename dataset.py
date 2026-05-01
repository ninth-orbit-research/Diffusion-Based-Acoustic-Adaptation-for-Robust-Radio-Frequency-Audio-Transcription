"""
Dataset handler for RF-degraded audio and paired transcripts.
Uses Whisper's built-in feature extraction for compatibility.
"""

import os
import torch
import numpy as np
from torch.utils.data import Dataset
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import torchaudio
from transformers import WhisperProcessor


class RFAudioDataset(Dataset):
    """
    Dataset for RF-degraded audio with paired transcripts.
    Uses Whisper's processor for feature extraction.
    
    Args:
        audio_dir: Directory containing .wav files
        transcript_dir: Directory containing .txt transcripts
        processor: WhisperProcessor instance (optional, will create if None)
        max_audio_length: Maximum audio length in seconds (default: 30)
    """
    
    def __init__(
        self,
        audio_dir: str,
        transcript_dir: str,
        processor: Optional[WhisperProcessor] = None,
        max_audio_length: float = 30.0,
        augment: bool = False
    ):
        self.audio_dir = Path(audio_dir)
        self.transcript_dir = Path(transcript_dir)
        self.max_audio_length = max_audio_length
        self.augment = False
        
        # Use provided processor or create new one
        if processor is None:
            self.processor = WhisperProcessor.from_pretrained(
                "openai/whisper-large-v3-turbo"
            )
        else:
            self.processor = processor
        
        # Whisper expects 16kHz audio
        self.sample_rate = 16000
        self.max_samples = int(self.max_audio_length * self.sample_rate)
        
        # Load dataset pairs
        self.samples = self._load_dataset_pairs()
        
        print(f"Loaded {len(self.samples)} audio-transcript pairs")
        print(f"Using Whisper's built-in feature extraction")
    
    def _load_dataset_pairs(self) -> List[Tuple[Path, str]]:
        """Load and validate audio-transcript pairs."""
        pairs = []
        audio_files = sorted(self.audio_dir.glob("*.mp3"))
        
        print(f"Scanning {len(audio_files)} audio files...")

        for audio_path in audio_files:
            transcript_path = self.transcript_dir / f"{audio_path.stem}.txt"

            if not transcript_path.exists():
                continue

            with open(transcript_path, 'r', encoding='utf-8') as f:
                transcript = f.read().strip()

            if not transcript:
                continue

            # 🔥 VALIDATE AUDIO HERE
            try:
                torchaudio.load(str(audio_path))
            except Exception as e:
                print(f"[SKIPPED - CORRUPTED] {audio_path.name} -> {e}")
                continue

            pairs.append((audio_path, transcript))

        print(f"Valid pairs after filtering: {len(pairs)}")
        return pairs

    def _load_audio(self, path: Path) -> np.ndarray:
        try:
            waveform, sample_rate = torchaudio.load(str(path))
        except Exception as e:
            raise RuntimeError(f"Audio decode failed: {path} | {e}")

        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        if sample_rate != self.sample_rate:
            resampler = torchaudio.transforms.Resample(
                orig_freq=sample_rate,
                new_freq=self.sample_rate
            )
            waveform = resampler(waveform)

        audio = waveform.squeeze().numpy()

        if len(audio) > self.max_samples:
            audio = audio[:self.max_samples]
        elif len(audio) < self.max_samples:
            audio = np.pad(audio, (0, self.max_samples - len(audio)))

        return audio


    def _augment_audio(self, audio: np.ndarray) -> np.ndarray:
        """
        Apply data augmentation to audio.
        
        Augmentations:
        - Amplitude scaling (0.8-1.2x)
        - Background noise injection
        - Simple time shift
        """
        if not self.augment:
            return audio
        
        # Random amplitude scaling
        if np.random.random() < 0.5:
            scale = np.random.uniform(0.8, 1.2)
            audio = audio * scale
        
        # Add random Gaussian noise
        if np.random.random() < 0.5:
            noise_factor = np.random.uniform(0.001, 0.005)
            noise = np.random.randn(len(audio)) * noise_factor
            audio = audio + noise
        
        # Random time shift (small)
        if np.random.random() < 0.3:
            shift = np.random.randint(-self.sample_rate // 10, self.sample_rate // 10)
            audio = np.roll(audio, shift)
        
        # Clip to prevent overflow
        audio = np.clip(audio, -1.0, 1.0)
        
        return audio
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict:
        """
        Get a single sample.
        
        Returns:
            Dictionary containing:
            - mel_spectrogram: (128, T) tensor from Whisper processor
            - transcript: string
            - audio_path: path to original audio file
        """
        audio_path, transcript = self.samples[idx]
        
        # Load audio
        audio = self._load_audio(audio_path)
        
        # Apply augmentation if enabled
        if self.augment:
            audio = self._augment_audio(audio)
        
        # Use Whisper's processor to extract features
        # This ensures compatibility with Whisper's expected input format
        inputs = self.processor(
            audio,
            sampling_rate=self.sample_rate,
            return_tensors="pt"
        )
        
        # Extract mel spectrogram (input_features)
        # Shape: (1, 128, 3000) -> squeeze to (128, 3000)
        mel_spectrogram = inputs.input_features.squeeze(0)
        
        return {
            'mel_spectrogram': mel_spectrogram,
            'transcript': transcript,
            'audio_path': str(audio_path)
        }


def collate_fn(batch):
    """
    Collate function for DataLoader.
    
    Since Whisper processor already handles padding/truncation,
    we just need to stack the tensors.
    """
    mel_specs = [item["mel_spectrogram"] for item in batch]
    transcripts = [item["transcript"] for item in batch]
    
    # Stack mel spectrograms (all should be same shape from processor)
    mel_tensor = torch.stack(mel_specs)
    
    return {
        "mel_spectrogram": mel_tensor,
        "transcript": transcripts
    }


# if __name__ == "__main__":
#     # Test dataset loading
#     import argparse
#     from transformers import WhisperProcessor
    
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--audio_dir', type=str, required=True)
#     parser.add_argument('--transcript_dir', type=str, required=True)
#     args = parser.parse_args()
    
#     # Create processor
#     processor = WhisperProcessor.from_pretrained("openai/whisper-large-v3-turbo")
    
#     # Create dataset
#     dataset = RFAudioDataset(
#         args.audio_dir, 
#         args.transcript_dir,
#         processor=processor
#     )
    
#     print(f"\nDataset size: {len(dataset)}")
    
#     # Test first sample
#     sample = dataset[0]
#     print(f"\nFirst sample:")
#     print(f"  Spectrogram shape: {sample['mel_spectrogram'].shape}")
#     print(f"  Expected shape: (128, 3000) for Whisper")
#     print(f"  Transcript: {sample['transcript'][:100]}...")
#     print(f"  Audio path: {sample['audio_path']}")
    
#     # Test collate function
#     from torch.utils.data import DataLoader
#     loader = DataLoader(dataset, batch_size=2, collate_fn=collate_fn)
#     batch = next(iter(loader))
#     print(f"\nBatch test:")
#     print(f"  Mel batch shape: {batch['mel_spectrogram'].shape}")
#     print(f"  Number of transcripts: {len(batch['transcript'])}")