# RF-Domain ASR: Technical Documentation

## System Architecture

The RF-Domain ASR system is a two-stage architecture designed to handle RF-degraded audio:

### Stage I: Parameter-Efficient Domain Adaptation (LoRA)

**Objective**: Adapt Whisper Large-V3-Turbo to RF-specific linguistic patterns while preserving general language understanding.

**Key Components**:
- **Foundation Model**: Whisper Large-V3-Turbo (809M parameters)
  - 32 encoder layers, 4 decoder layers (distilled from 32)
  - 3.8× faster inference vs. Whisper Large-V3
  - 96.5% accuracy retention

- **LoRA Configuration**:
  - Rank (r): 64
  - Scaling factor (α): 32
  - Target modules: Query, Key, Value, Output projections
  - Trainable parameters: 8.2M (1% of total)

- **Training Protocol**:
  - Optimizer: AdamW
  - Learning rate: 3×10⁻⁴ with cosine annealing
  - Warmup: 100 steps
  - Batch size: 4 (with gradient accumulation)
  - FP16 mixed precision
  - Early stopping: patience = 4 epochs

**Performance**: 42% relative WER reduction over zero-shot baseline

### Stage II: Diffusion-Inspired Acoustic Refinement

**Objective**: Transform RF-degraded spectrograms into cleaner representations using gradient signals from frozen Whisper.

**Key Components**:

1. **U-Net Architecture** (31M parameters):
   - Encoder: 4 downsampling blocks (64→128→256→512→1024 channels)
   - Bottleneck: 1024 channels
   - Decoder: 4 upsampling blocks with skip connections
   - Timestep conditioning via sinusoidal embeddings

2. **Iterative Refinement**:
   ```
   x_t = x_{t-1} - U_θ(x_{t-1}, t)  for t = 1, 2, 3, 4
   ```
   - T = 4 refinement steps (optimal balance)
   - Each step subtracts predicted noise residual

3. **Training Strategy**:
   - **Identity Initialization**: Train U-Net to predict zero initially
   - **End-to-End Gradient Flow**: Backprop through all refinement steps
   - **Margin Regularization**: Enforce monotonic confidence improvement

4. **Loss Function**:
   ```
   L_total = L_CE + λ · Σ_t L_margin^(t)
   ```
   where:
   - L_CE: Cross-entropy ASR loss on final refinement
   - L_margin: Hinge loss on logit margins
   - λ = 0.5 (balancing coefficient)

**Performance**: Additional 40% relative WER reduction over LoRA-only

### Complete Pipeline Performance

| Configuration | WER (%) | Relative Improvement |
|--------------|---------|---------------------|
| Zero-shot Whisper | 43.6 | Baseline |
| LoRA-adapted (Stage I) | 25.4 | 42% |
| Full pipeline (Stage I+II) | 15.2 | 65% |

## Theoretical Framework

### 1. Implicit Prior Distillation

The refinement network learns by maximizing:
```
J(θ) = E_{(x_0, y) ~ D_RF}[log p_Whisper(y | f_θ(x_0))]
```

This objective implicitly projects RF-degraded inputs onto the pretraining distribution through the frozen Whisper model's conditional likelihood.

**Key Insight**: The frozen Whisper encodes an implicit acoustic density model from 680,000 hours of training. By optimizing refinement to maximize transcript likelihood, we distill this knowledge without requiring paired clean-noisy data.

### 2. Gradient Flow Through Refinement Chain

Gradients from ASR loss propagate through all T steps:
```
∂L_CE/∂θ = Σ_t (∂L_CE/∂x_T · Π_{k=t}^{T-1} ∂x_{k+1}/∂x_k · ∂x_t/∂θ)
```

This enables coordinated learning across iterations, with earlier steps focusing on coarse denoising and later steps on fine-grained refinement.

### 3. Margin Regularization as Curriculum Learning

The margin loss:
```
L_margin^(t) = (1/N) Σ_i max(0, m_{t-1}^(i) - m_t^(i) + ε)
```

enforces that prediction confidence increases monotonically, implementing implicit curriculum learning where each iteration builds on previous improvements.

## Implementation Details

### Preprocessing Pipeline

1. **Audio Loading**:
   - Resample to 16kHz using polyphase filtering (kaiser_best)
   - Convert stereo to mono
   - Pad/truncate to max_length (default: 30s)

2. **Mel Spectrogram Extraction**:
   - STFT: n_fft=400 (25ms), hop=160 (10ms)
   - Hanning window
   - 80 mel bins, 0-8000 Hz
   - Log compression: log(mel_spec + 1e-10)

### Training Hyperparameters

**Stage I (LoRA)**:
- Learning rate: 3×10⁻⁴
- Weight decay: 1×10⁻²
- Gradient clipping: max_norm = 1.0
- Batch size: 4 (effective 16 with accumulation)
- Training time: ~2 hours (A100 GPU)

**Stage II (Refinement)**:
- Identity init: 2000-3000 steps until L < 10⁻⁴
- Learning rate: 5×10⁻⁴
- Weight decay: 1×10⁻³
- Margin weight: 0.5
- Margin slack: 0.1
- Batch size: 8
- Training time: ~6 hours (A100 GPU)

### Inference Latency

Per 30-second audio clip on A100:
- LoRA-adapted Whisper: 0.58s
- Full pipeline (T=4): 1.00s
- Speedup vs Whisper Large-V3: 3.8×

## Key Design Decisions

### 1. Why LoRA over Full Fine-Tuning?

**Problem**: 500 samples × 4 hours insufficient for 809M parameters
- Full fine-tuning → catastrophic overfitting
- Training/validation WER divergence within 5 epochs
- Loss of clean speech transcription ability

**Solution**: Low-rank constraint provides implicit regularization
- Updates constrained to rank-64 subspace
- Preserves pretrained knowledge in orthogonal directions
- 10× parameter reduction per attention matrix

### 2. Why T=4 Refinement Steps?

Ablation study results:
- T=1: 21.8% WER (underrefinement)
- T=2: 17.3% WER (improving)
- **T=4: 15.2% WER** (optimal)
- T=8: 14.9% WER (diminishing returns, 2× slower)

Optimal balance between performance and computational cost.

### 3. Why Identity Initialization?

Without identity init:
- 73% of training runs diverge (11/15)
- Random initialization → arbitrary noise predictions
- Whisper receives corrupted inputs → infinite loss

With identity init:
- All runs converge successfully
- Smooth transition from x₁ ≈ x₀ to refined outputs
- Stable gradient flow from initialization

### 4. Why Margin Regularization?

Without margin loss (λ=0):
- 22% WER increase (15.2% → 18.5%)
- Confidence oscillates across iterations
- Network makes destructive transformations

With margin loss (λ=0.5):
- Monotonic confidence progression
- Stable refinement trajectory
- Better final performance

## Extensions and Future Work

### 1. Multi-SNR Training
Currently trained on mixed SNR (0-20 dB). Could stratify by SNR condition:
- Low SNR specialist (0-5 dB)
- Medium SNR specialist (5-10 dB)
- High SNR specialist (10-20 dB)

### 2. Adaptive Refinement Steps
Learn when to stop refinement based on confidence:
- Exit criterion: margin > threshold
- Reduces inference cost for cleaner inputs
- Maintains quality on degraded inputs

### 3. Multi-Speaker Support
Current system trained on single-speaker RF audio. Extensions:
- Speaker diarization module
- Multi-speaker refinement network
- Cross-talk suppression

### 4. Real-Time Streaming
Current system processes fixed-length segments. For streaming:
- Overlapping window processing
- State preservation across windows
- Latency-optimized refinement (T=2)

## Troubleshooting

### Common Issues

**1. Out of Memory (OOM)**
```bash
# Reduce batch size
--batch_size 2

# Use gradient checkpointing
# (Add to train_refinement.py if needed)
```

**2. Training Divergence**
```bash
# Lower learning rate
--learning_rate 1e-4

# Increase warmup
--warmup_steps 500

# Check data quality
python dataset.py --audio_dir data/audio --transcript_dir data/transcripts
```

**3. Poor Performance**
- Verify data quality and alignment
- Check SNR distribution (should cover 0-20 dB)
- Ensure sufficient training samples (>100)
- Validate LoRA model before refinement training

## Citation

If you use this implementation, please cite:

```bibtex
@article{rf_asr_2024,
  title={Robust ASR in RF-Degraded Environments via LoRA Adaptation 
         and Diffusion-Inspired Acoustic Refinement},
  year={2024}
}
```

## License

This implementation is provided for research purposes. Whisper models are subject to OpenAI's license terms.
