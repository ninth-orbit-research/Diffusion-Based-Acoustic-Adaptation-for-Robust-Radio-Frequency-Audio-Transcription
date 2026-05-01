# RF-Domain ASR Implementation - File Structure

## Core Implementation Files

### 1. Dataset Management
- **`dataset.py`**: RFAudioDataset class for loading audio/transcript pairs
  - Handles audio preprocessing (resampling, mel spectrogram conversion)
  - Implements data augmentation (optional)
  - Custom collate function for variable-length sequences

### 2. Model Architectures
- **`unet.py`**: U-Net refinement network
  - Encoder-decoder with skip connections
  - Timestep conditioning via sinusoidal embeddings
  - ~31M parameters

### 3. Training Scripts

#### Stage I: LoRA Adaptation
- **`train_lora.py`**: LoRA fine-tuning of Whisper
  - Parameter-efficient adaptation (1% of parameters)
  - Early stopping and checkpoint averaging
  - Validation-based model selection

#### Stage II: Acoustic Refinement
- **`train_refinement.py`**: U-Net refinement training
  - Identity initialization phase
  - Iterative refinement with margin regularization
  - End-to-end gradient flow through T=4 steps

### 4. Inference and Evaluation
- **`inference.py`**: Complete inference pipeline
  - Single file or batch processing
  - Optional baseline comparison
  - Configurable refinement steps

- **`evaluate.py`**: Comprehensive evaluation suite
  - Zero-shot, LoRA-only, and full pipeline comparison
  - Ablation studies on refinement steps
  - Automatic result visualization

### 5. Utilities
- **`generate_demo_data.py`**: Synthetic RF data generation
  - Simulates RF degradation (noise, bandwidth limitation, distortion)
  - Creates paired audio/transcript samples
  - Useful for testing and demonstration

- **`run_pipeline.sh`**: End-to-end automation script
  - Runs complete training and evaluation pipeline
  - Generates demo data if needed
  - Provides usage examples

### 6. Documentation
- **`README.md`**: Quick start guide and overview
- **`TECHNICAL_DOCS.md`**: Detailed technical documentation
- **`requirements.txt`**: Python dependencies

## Directory Structure (After Running)

```
rf-asr-implementation/
├── dataset.py                  # Dataset loader
├── unet.py                     # U-Net architecture
├── train_lora.py              # Stage I training
├── train_refinement.py        # Stage II training
├── inference.py               # Inference script
├── evaluate.py                # Evaluation suite
├── generate_demo_data.py      # Demo data generator
├── run_pipeline.sh            # Automation script
├── requirements.txt           # Dependencies
├── README.md                  # Quick start
├── TECHNICAL_DOCS.md          # Technical docs
│
├── data/                      # Dataset directory
│   └── demo/                  # Demo data (generated)
│       ├── audio/             # .wav files
│       ├── transcripts/       # .txt files
│       └── dataset_info.json
│
└── outputs/                   # Training outputs
    ├── lora/                  # Stage I models
    │   ├── best_model/
    │   └── checkpoint_epoch_*/
    ├── refinement/            # Stage II models
    │   └── best_model/
    │       └── unet_checkpoint.pt
    └── evaluation/            # Evaluation results
        ├── evaluation_results.json
        ├── performance_comparison.png
        └── ablation_results.json
```

## Quick Start

### 1. Installation
```bash
pip install -r requirements.txt
```

### 2. Generate Demo Data
```bash
python generate_demo_data.py --output_dir data/demo --num_samples 100
```

### 3. Run Complete Pipeline
```bash
bash run_pipeline.sh
```

Or run stages individually:

### 4. Stage I: Train LoRA Adapter
```bash
python train_lora.py \
  --audio_dir data/demo/audio \
  --transcript_dir data/demo/transcripts \
  --output_dir outputs/lora \
  --epochs 12 \
  --batch_size 4
```

### 5. Stage II: Train Refinement Network
```bash
python train_refinement.py \
  --audio_dir data/demo/audio \
  --transcript_dir data/demo/transcripts \
  --lora_model outputs/lora/best_model \
  --output_dir outputs/refinement \
  --epochs 50 \
  --refinement_steps 4
```

### 6. Inference
```bash
python inference.py \
  --audio_file test.wav \
  --lora_model outputs/lora/best_model \
  --refinement_model outputs/refinement/best_model/unet_checkpoint.pt \
  --output_file transcript.txt
```

### 7. Evaluation
```bash
python evaluate.py \
  --audio_dir data/demo/audio \
  --transcript_dir data/demo/transcripts \
  --lora_model outputs/lora/best_model \
  --refinement_model outputs/refinement/best_model/unet_checkpoint.pt \
  --output_dir outputs/evaluation
```

## Key Features by File

### dataset.py
✓ Automatic audio resampling to 16kHz
✓ Log-mel spectrogram extraction (80 channels)
✓ Paired audio-transcript loading
✓ Optional data augmentation
✓ Handles variable-length sequences

### unet.py
✓ Standard U-Net with skip connections
✓ Timestep conditioning for iterative refinement
✓ ~31M parameters
✓ Predicts noise residuals for subtraction

### train_lora.py
✓ LoRA rank 64, alpha 32
✓ Selective adaptation of attention matrices
✓ 8.2M trainable parameters (1% of Whisper)
✓ AdamW optimizer with cosine annealing
✓ Early stopping and checkpoint management

### train_refinement.py
✓ Identity initialization (critical for stability)
✓ Iterative refinement (T=4 steps)
✓ Margin regularization for monotonic improvement
✓ End-to-end gradient flow
✓ Frozen LoRA-adapted Whisper as discriminator

### inference.py
✓ Single file and batch processing
✓ Baseline comparison mode
✓ Configurable refinement steps
✓ Clean output formatting

### evaluate.py
✓ Multi-configuration comparison
✓ WER and CER metrics
✓ Ablation studies
✓ Automatic visualization
✓ JSON result export

## Configuration Examples

### For Real RF Data
Replace demo data generation with your own data:
```
data/
├── audio/
│   ├── recording_001.wav
│   ├── recording_002.wav
│   └── ...
└── transcripts/
    ├── recording_001.txt
    ├── recording_002.txt
    └── ...
```

Then run training with your data paths:
```bash
python train_lora.py \
  --audio_dir data/audio \
  --transcript_dir data/transcripts
```

### For Different Model Sizes
Modify LoRA configuration:
```bash
python train_lora.py \
  --lora_rank 32 \    # Smaller rank (faster, less capacity)
  --lora_alpha 16     # Adjust scaling accordingly
```

### For Different Refinement Configurations
```bash
python train_refinement.py \
  --refinement_steps 2 \     # Faster inference
  --margin_weight 1.0 \      # Stronger regularization
  --learning_rate 1e-3       # Faster convergence
```

## Performance Expectations

With 100 training samples (~4 hours of audio):
- **LoRA training**: ~2 hours on A100
- **Refinement training**: ~6 hours on A100
- **Inference**: ~1 second per 30-second clip

Expected WER improvements:
- Zero-shot → LoRA: ~40% relative reduction
- LoRA → Full pipeline: ~40% additional reduction
- Overall: ~65% relative improvement

## Troubleshooting Guide

### Issue: CUDA Out of Memory
**Solution**: Reduce batch size
```bash
--batch_size 2  # or --batch_size 1
```

### Issue: Training Diverges
**Solution**: Lower learning rate
```bash
--learning_rate 1e-4
```

### Issue: Poor Performance
**Checklist**:
1. Verify data quality (check a few samples manually)
2. Ensure audio-transcript alignment
3. Check SNR distribution covers target range
4. Increase training samples if possible

## Dependencies

Core requirements:
- PyTorch >= 2.0.0
- Transformers >= 4.30.0
- PEFT >= 0.4.0 (for LoRA)
- Librosa >= 0.10.0 (audio processing)
- jiwer >= 3.0.0 (WER computation)

See `requirements.txt` for complete list.

## Next Steps

1. **Test with demo data**: Run `bash run_pipeline.sh`
2. **Prepare your RF data**: Follow the directory structure
3. **Train on your data**: Use the training scripts
4. **Evaluate results**: Use evaluate.py
5. **Deploy inference**: Use inference.py

For detailed technical information, see `TECHNICAL_DOCS.md`.
