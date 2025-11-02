#!/bin/bash
# Optimized PK-YOLO Training for Small Tumor Detection
# Output structure:
#   experiments/
#     ├─ spark_pretrain/<YYYYmmdd_HHMMSS>/
#     └─ pkyolo_train/<YYYYmmdd_HHMMSS>/

set -euo pipefail

# ---------------- Configuration ----------------
DATA_DIR="./data"
EXPERIMENTS_DIR="./experiments"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# Folder layout (with nested timestamp folder)
SPARK_OUTPUT="${EXPERIMENTS_DIR}/spark_pretrain/${TIMESTAMP}"
TRAIN_OUTPUT="${EXPERIMENTS_DIR}/pkyolo_train/${TIMESTAMP}"

# Training parameters optimized for small tumors
SPARK_EPOCHS=100
SPARK_BATCH_SIZE=32
SPARK_LR=0.001
PRETRAIN_IMG=320

TRAIN_EPOCHS=150
TRAIN_BATCH_SIZE=16       # Smaller batch for better gradients
TRAIN_LR=5e-4             # Higher initial LR
BACKBONE_LR_MULT=0.05     # Lower multiplier for pretrained backbone
IMG_SIZE=640
WORKERS=16

# ---------------- Prep ----------------
# Validate data
if [ ! -d "$DATA_DIR" ]; then
  echo "ERROR: Data directory not found: $DATA_DIR" >&2
  exit 1
fi
if [ ! -d "$DATA_DIR/train" ]; then
  echo "ERROR: Training split not found: $DATA_DIR/train" >&2
  exit 1
fi

# Create directories
mkdir -p "$SPARK_OUTPUT" "$TRAIN_OUTPUT"

echo "========================================================================"
echo "PK-YOLO Training Pipeline (Optimized for Small Tumors)"
echo "========================================================================"
echo "Timestamp:             $TIMESTAMP"
echo "Data directory:        $DATA_DIR"
echo "SparK output:          $SPARK_OUTPUT"
echo "Training output:       $TRAIN_OUTPUT"
echo "========================================================================"

# ---------------- Stage 1: SparK Pretraining (optional) ----------------
SPARK_WEIGHTS=""
if [ "${1:-}" = "--with-spark" ] || [ "${1:-}" = "" ]; then
  echo ""
  echo "[Stage 1/2] Starting SparK Pretraining..."
  echo "Parameters:"
  echo "  - Epochs:       $SPARK_EPOCHS"
  echo "  - Batch size:   $SPARK_BATCH_SIZE"
  echo "  - LR:           $SPARK_LR"
  echo "  - Image size:   $PRETRAIN_IMG"
  echo ""

  python3 spark_pretrain.py \
    --data-dir "$DATA_DIR" \
    --out-dir "$SPARK_OUTPUT" \
    --epochs "$SPARK_EPOCHS" \
    --batch-size "$SPARK_BATCH_SIZE" \
    --lr "$SPARK_LR" \
    --opt adamw \
    --patch 16 \
    --mask-ratio 0.75 \
    --img-size "$PRETRAIN_IMG" \
    --patience 15 \
    --min-delta 1e-4 \
    --workers "$WORKERS" \
    --amp

  # Prefer best checkpoint; fallback to latest epoch
  if [ -f "${SPARK_OUTPUT}/best_spark_model.pth" ]; then
    SPARK_WEIGHTS="${SPARK_OUTPUT}/best_spark_model.pth"
  else
    SPARK_WEIGHTS=$(ls -t "${SPARK_OUTPUT}"/repvit_spark_epoch*.pt 2>/dev/null | head -1 || true)
    if [ -z "${SPARK_WEIGHTS:-}" ]; then
      echo "ERROR: No SparK weights produced in ${SPARK_OUTPUT}" >&2
      exit 1
    fi
    echo "WARNING: best_spark_model.pth not found; using latest checkpoint: $SPARK_WEIGHTS"
  fi
  echo "Using SparK weights: $SPARK_WEIGHTS"

elif [ "${1:-}" = "--no-spark" ]; then
  echo "Skipping SparK pretraining as requested (--no-spark)."
  SPARK_WEIGHTS=""
else
  echo "Usage: $0 [--with-spark|--no-spark]"
  echo "Default: --with-spark"
  exit 1
fi

# ---------------- Stage 2: PK-YOLO Training ----------------
echo ""
echo "[Stage 2/2] Starting PK-YOLO Training..."
echo "Parameters:"
echo "  - Epochs:               $TRAIN_EPOCHS"
echo "  - Batch size:           $TRAIN_BATCH_SIZE"
echo "  - LR:                   $TRAIN_LR"
echo "  - Backbone LR mult:     $BACKBONE_LR_MULT"
echo "  - Image size:           $IMG_SIZE"
if [ -n "$SPARK_WEIGHTS" ]; then
  echo "  - Pretrained backbone:  $SPARK_WEIGHTS"
fi
echo ""

python3 training_script.py \
  --data_dir "$DATA_DIR" \
  --output_dir "$TRAIN_OUTPUT" \
  --epochs "$TRAIN_EPOCHS" \
  --batch_size "$TRAIN_BATCH_SIZE" \
  --lr "$TRAIN_LR" \
  --backbone_lr_mult "$BACKBONE_LR_MULT" \
  --img_size "$IMG_SIZE" \
  --workers "$WORKERS" \
  --mixed_precision \
  ${SPARK_WEIGHTS:+--spark_backbone_path "$SPARK_WEIGHTS"}

# ---------------- Footer ----------------
echo ""
echo "========================================================================"
echo "PIPELINE COMPLETED SUCCESSFULLY"
echo "========================================================================"
echo "Results:"
if [ -n "$SPARK_WEIGHTS" ]; then
  echo "  - SparK backbone:      $SPARK_WEIGHTS"
fi
echo "  - PK-YOLO checkpoint:  $TRAIN_OUTPUT/best_model.pth"
echo "  - SparK folder:        $SPARK_OUTPUT/"
echo "  - Training folder:     $TRAIN_OUTPUT/"
echo "========================================================================"
