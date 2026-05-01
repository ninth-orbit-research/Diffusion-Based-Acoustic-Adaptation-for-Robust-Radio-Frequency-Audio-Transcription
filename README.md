# RF-Domain ASR: LoRA Adaptation + Diffusion-Inspired Acoustic Refinement

Implementation of the two-stage architecture for robust automatic speech recognition (ASR) in radio frequency (RF)-degraded environments.

## Overview

This system employs:
1. **Stage I**: Parameter-efficient domain adaptation using Low-Rank Adaptation (LoRA)
2. **Stage II**: Diffusion-inspired acoustic refinement with iterative U-Net denoising

## Architecture

```
Raw RF Audio → Log-Mel Spectrogram → LoRA-Adapted Whisper → Iterative U-Net Refinement → Transcription
```

### Stage I: LoRA Adaptation
- Foundation model: Whisper Large-V3-Turbo
- LoRA rank: 64, scaling factor: 32
- Applied to attention projection matrices only
- Achieves 42% relative WER improvement

### Stage II: Acoustic Refinement
- U-Net architecture with skip connections
- 4 iterative refinement steps (T=4)
- Identity initialization for stability
- Margin regularization for monotonic improvement
- Additional 40% relative WER improvement

## Dataset Structure

```
data/
├── audio/          # RF-degraded audio files (.wav)
│   ├── sample_001.wav
│   ├── sample_002.wav
│   └── ...
└── transcripts/    # Paired transcripts (.txt)
    ├── sample_001.txt
    ├── sample_002.txt
    └── ...
```

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### 1. Prepare Dataset
Place your audio files in `data/audio/` and corresponding transcripts in `data/transcripts/` with matching filenames.

### 2. Stage I: Train LoRA Adapter
```bash
python train_lora.py \
    --audio_dir data/audio \
    --transcript_dir data/transcripts \
    --output_dir outputs/lora \
    --epochs 12 \
    --batch_size 4 \
    --learning_rate 3e-4
```

### 3. Stage II: Train Refinement Network
```bash
python train_refinement.py \
    --audio_dir data/audio \
    --transcript_dir data/transcripts \
    --lora_model outputs/lora/best_model \
    --output_dir outputs/refinement \
    --epochs 50 \
    --refinement_steps 4
```

### 4. Inference
```bash
python inference.py \
    --audio_file test_audio.wav \
    --lora_model outputs/lora/best_model \
    --refinement_model outputs/refinement/best_model.pt \
    --output_file transcript.txt
```

## Key Features

- **Parameter Efficiency**: Only 8.2M LoRA parameters (1% of Whisper)
- **No Paired Clean Data Required**: Uses transcript supervision instead
- **Gradient Flow**: End-to-end backpropagation through refinement steps
- **Stability**: Identity initialization + margin regularization
- **Real-time Capable**: 3.8× faster than Whisper Large-V3

## Performance

| SNR Condition | Zero-Shot | LoRA Only | Full Pipeline | Improvement |
|---------------|-----------|-----------|---------------|-------------|
| Clean (>20 dB) | 8.7% | 4.2% | 3.1% | 64% |
| High (10-20 dB) | 32.4% | 18.3% | 11.7% | 64% |
| Medium (5-10 dB) | 58.1% | 31.8% | 19.4% | 67% |
| Low (0-5 dB) | 75.2% | 46.7% | 26.8% | 64% |
| **Overall** | **43.6%** | **25.4%** | **15.2%** | **65%** |
