#!/bin/bash
set -e

echo "=============================================="
echo " TempoRAL Expert Distill - Docker Entrypoint"
echo " Target: RTX 4090 (24GB) + 128GB RAM"
echo "=============================================="

# GPU info
if command -v nvidia-smi &> /dev/null; then
    echo ""
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
    echo ""
fi

# ---------------------------------------------------------------------------
# 1. Download dataset and checkpoint if not already present
# ---------------------------------------------------------------------------
DATASET_DIR="${DATASET_DIR:-/dataset/droid_100}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-/checkpoint/pi05_droid}"

if [ ! -d "$DATASET_DIR" ] || [ -z "$(ls -A $DATASET_DIR 2>/dev/null)" ]; then
    echo "[1/2] Downloading DROID droid_100 dataset..."
    mkdir -p "$DATASET_DIR"
    gsutil -m cp -r gs://gresearch/robotics/droid_100/* "$DATASET_DIR/"
    echo "  -> Dataset downloaded to $DATASET_DIR"
else
    echo "[1/2] DROID dataset already exists at $DATASET_DIR, skipping download."
fi

if [ ! -d "$CHECKPOINT_DIR" ] || [ -z "$(ls -A $CHECKPOINT_DIR 2>/dev/null)" ]; then
    echo "[2/2] Downloading pi0.5-DROID checkpoint..."
    mkdir -p "$CHECKPOINT_DIR"
    gsutil -m cp -r gs://openpi-assets/checkpoints/pi05_droid/* "$CHECKPOINT_DIR/"
    echo "  -> Checkpoint downloaded to $CHECKPOINT_DIR"
else
    echo "[2/2] pi0.5-DROID checkpoint already exists at $CHECKPOINT_DIR, skipping download."
fi

echo ""
echo "Dataset size: $(du -sh $DATASET_DIR | cut -f1)"
echo "Checkpoint size: $(du -sh $CHECKPOINT_DIR | cut -f1)"
echo ""

# ---------------------------------------------------------------------------
# 2. Run training
# ---------------------------------------------------------------------------
echo "=============================================="
echo " Starting Expert Distill Training"
echo "=============================================="

python /app/run_expert_distill.py \
    --checkpoint-dir "$CHECKPOINT_DIR" \
    --dataset-dir "$DATASET_DIR" \
    --output-dir "${OUTPUT_DIR:-/output}" \
    --distill-steps "${DISTILL_STEPS:-5000}" \
    --batch-size "${BATCH_SIZE:-16}" \
    --lr "${LR:-1e-3}" \
    --alpha "${ALPHA:-0.05}" \
    --subsequence-len "${SUBSEQUENCE_LEN:-50}"

echo ""
echo "=============================================="
echo " Training complete! Results in ${OUTPUT_DIR:-/output}/"
echo "=============================================="
ls -la "${OUTPUT_DIR:-/output}/"
