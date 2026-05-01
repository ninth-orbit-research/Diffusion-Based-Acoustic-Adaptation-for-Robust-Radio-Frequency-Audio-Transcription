#!/bin/bash

# RF-Domain ASR: Complete Training and Evaluation Pipeline
# This script demonstrates the full workflow from data generation to evaluation

set -e  # Exit on error

echo "=========================================="
echo "RF-Domain ASR - Complete Pipeline"
echo "=========================================="
echo ""

# Configuration
DATA_DIR="data/demo"
OUTPUT_DIR="outputs"
LORA_DIR="${OUTPUT_DIR}/lora"
REFINEMENT_DIR="${OUTPUT_DIR}/refinement"
EVAL_DIR="${OUTPUT_DIR}/evaluation"

# Step 1: Generate demo data (if not exists)
if [ ! -d "$DATA_DIR/audio" ]; then
    echo "Step 1: Generating demo dataset..."
    python generate_demo_data.py \
        --output_dir $DATA_DIR \
        --num_samples 100 \
        --min_snr 0 \
        --max_snr 20
    echo ""
else
    echo "Step 1: Demo data already exists, skipping generation"
    echo ""
fi

# Step 2: Stage I - Train LoRA adapter
echo "Step 2: Training LoRA adapter (Stage I)..."
echo "This adapts Whisper to RF-degraded audio"
echo ""

python train_lora.py \
    --audio_dir ${DATA_DIR}/audio \
    --transcript_dir ${DATA_DIR}/transcripts \
    --output_dir $LORA_DIR \
    --epochs 12 \
    --batch_size 1 \
    --learning_rate 3e-4 \
    --lora_rank 64 \
    --lora_alpha 32 \
    --patience 4

echo ""
echo "✓ LoRA training complete!"
echo ""

# Step 3: Stage II - Train refinement network
echo "Step 3: Training refinement network (Stage II)..."
echo "This learns to iteratively refine spectrograms"
echo ""

python train_refinement.py \
    --audio_dir ${DATA_DIR}/audio \
    --transcript_dir ${DATA_DIR}/transcripts \
    --lora_model ${LORA_DIR}/best_model \
    --output_dir $REFINEMENT_DIR \
    --epochs 50 \
    --batch_size 1 \
    --refinement_steps 4 \
    --learning_rate 5e-4 \
    --margin_weight 0.5 \
    --patience 5

echo ""
echo "✓ Refinement training complete!"
echo ""

# Step 4: Evaluation
echo "Step 4: Running comprehensive evaluation..."
echo ""

python evaluate.py \
    --audio_dir ${DATA_DIR}/audio \
    --transcript_dir ${DATA_DIR}/transcripts \
    --lora_model ${LORA_DIR}/best_model \
    --refinement_model ${REFINEMENT_DIR}/best_model/unet_checkpoint.pt \
    --output_dir $EVAL_DIR \
    --batch_size 8 \
    --ablation

echo ""
echo "✓ Evaluation complete!"
echo ""

# Step 5: Test inference on a single file
echo "Step 5: Testing inference on sample file..."
echo ""

TEST_FILE="${DATA_DIR}/audio/sample_001.wav"
OUTPUT_TRANSCRIPT="outputs/test_transcript.txt"

python inference.py \
    --audio_file $TEST_FILE \
    --lora_model ${LORA_DIR}/best_model \
    --refinement_model ${REFINEMENT_DIR}/best_model/unet_checkpoint.pt \
    --output_file $OUTPUT_TRANSCRIPT \
    --show_baseline

echo ""
echo "=========================================="
echo "Pipeline Complete!"
echo "=========================================="
echo ""
echo "Results:"
echo "  - LoRA model: ${LORA_DIR}/best_model"
echo "  - Refinement model: ${REFINEMENT_DIR}/best_model"
echo "  - Evaluation results: ${EVAL_DIR}/evaluation_results.json"
echo "  - Performance plots: ${EVAL_DIR}/*.png"
echo "  - Test transcript: ${OUTPUT_TRANSCRIPT}"
echo ""
echo "To transcribe your own audio files:"
echo "  python inference.py \\"
echo "    --audio_file YOUR_FILE.wav \\"
echo "    --lora_model ${LORA_DIR}/best_model \\"
echo "    --refinement_model ${REFINEMENT_DIR}/best_model/unet_checkpoint.pt \\"
echo "    --output_file output.txt"
echo ""
