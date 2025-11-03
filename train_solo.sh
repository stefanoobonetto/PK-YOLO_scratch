#!/bin/bash
# PK-YOLO training only (uses a fixed SparK backbone)

set -euo pipefail

# ---------------- Configuration ----------------
DATA_DIR="./data"
EXPERIMENTS_DIR="./experiments"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

TRAIN_OUTPUT="${EXPERIMENTS_DIR}/pkyolo_train/${TIMESTAMP}"

# Training parameters (same as train.sh)
TRAIN_EPOCHS=150
TRAIN_BATCH_SIZE=16
TRAIN_LR=2e-4
BACKBONE_LR_MULT=0.05
IMG_SIZE=640
WORKERS=16

# SparK backbone (fixed as requested)
BACKBONE_PATH="experiments/spark_pretrain/20251102_195636/best_spark_model.pth"

# ---------------- Prep ----------------
if [ ! -d "$DATA_DIR" ]; then
  echo "ERROR: Data directory not found: $DATA_DIR" >&2
  exit 1
fi
if [ ! -d "$DATA_DIR/train" ]; then
  echo "ERROR: Training split not found: $DATA_DIR/train" >&2
  exit 1
fi
if [ ! -f "$BACKBONE_PATH" ]; then
  echo "ERROR: Backbone weights not found: $BACKBONE_PATH" >&2
  exit 1
fi

mkdir -p "$TRAIN_OUTPUT"

echo "========================================================================"
echo "PK-YOLO Training (ONLY) — Optimized for Small Tumors"
echo "========================================================================"
echo "Timestamp:             $TIMESTAMP"
echo "Data directory:        $DATA_DIR"
echo "Training output:       $TRAIN_OUTPUT"
echo "Backbone weights:      $BACKBONE_PATH"
echo "------------------------------------------------------------------------"
echo "Parameters:"
echo "  - Epochs:               $TRAIN_EPOCHS"
echo "  - Batch size:           $TRAIN_BATCH_SIZE"
echo "  - LR:                   $TRAIN_LR"
echo "  - Backbone LR mult:     $BACKBONE_LR_MULT"
echo "  - Image size:           $IMG_SIZE"
echo "  - Workers:              $WORKERS"
echo "========================================================================"
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
  --spark_backbone_path "$BACKBONE_PATH" \
  --save_visuals --vis_interval 200 --vis_conf 0.5 \
#   --mixed_precision \
echo ""
echo "========================================================================"
echo "PK-YOLO TRAINING COMPLETED"
echo "========================================================================"
echo "Results:"
echo "  - PK-YOLO checkpoint:  $TRAIN_OUTPUT/best_model.pth"
echo "  - Training folder:     $TRAIN_OUTPUT/"
echo "  - Backbone used:       $BACKBONE_PATH"
echo "========================================================================"
