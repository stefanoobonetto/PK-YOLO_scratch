#!/bin/bash
# PK-YOLO Training Pipeline: SparK Pretraining + Full Training

set -euo pipefail

# Data paths
DATA_DIR="./data"               # deve contenere ./train e (opzionale) ./val
OUTPUT_DIR="./experiments"

# Image sizes
IMG_SIZE=640                    # fine-tuning PK-YOLO
PRETRAIN_IMG=320                # pretraining SparK

# CPU workers
WORKERS=16

# ----------------------------
# SparK Pretraining Parameters
# ----------------------------
SPARK_EPOCHS=300
SPARK_BATCH_SIZE=16
SPARK_LR=0.001
SPARK_PATCH_SIZE=16
SPARK_MASK_RATIO=0.75           # più alto = più veloce (consigliato per MAE/SparK)

# Early Stopping (attivo solo se esiste $DATA_DIR/val)
SPARK_PATIENCE=15               # epoche senza miglioramento su val
SPARK_MIN_DELTA=1e-4            # miglioramento minimo su val per azzerare la pazienza

# ----------------------------
# PK-YOLO Training Parameters
# ----------------------------
TRAIN_EPOCHS=300
TRAIN_BATCH_SIZE=32
TRAIN_LR=0.0001
BACKBONE_LR_MULT=0.1

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
SPARK_OUTPUT="${OUTPUT_DIR}/spark_pretrain_${TIMESTAMP}"
TRAIN_OUTPUT="${OUTPUT_DIR}/pkyolo_train_${TIMESTAMP}"

mkdir -p "$SPARK_OUTPUT" "$TRAIN_OUTPUT"

echo "========================================================================"
echo "PK-YOLO Training Pipeline"
echo "========================================================================"
echo "Data directory:        $DATA_DIR"
echo "SparK output:          $SPARK_OUTPUT"
echo "Training output:       $TRAIN_OUTPUT"
echo "========================================================================"

# ==========================
# Stage 1: SparK Pretraining
# ==========================
echo ""
echo "[Stage 1/2] Starting SparK Pretraining..."
echo "Parameters:"
echo "  - Epochs:       $SPARK_EPOCHS  (early stop: $SPARK_PATIENCE / Δ>$SPARK_MIN_DELTA)"
echo "  - Batch size:   $SPARK_BATCH_SIZE"
echo "  - LR:           $SPARK_LR"
echo "  - Mask ratio:   $SPARK_MASK_RATIO"
echo "  - Patch size:   $SPARK_PATCH_SIZE"
echo "  - Img size:     $PRETRAIN_IMG"
echo "  - Workers:      $WORKERS"
echo ""

python3 spark_pretrain.py \
  --data-dir "$DATA_DIR" \
  --split train \
  --out-dir "$SPARK_OUTPUT" \
  --epochs "$SPARK_EPOCHS" \
  --batch-size "$SPARK_BATCH_SIZE" \
  --lr "$SPARK_LR" \
  --opt adamw \
  --patch "$SPARK_PATCH_SIZE" \
  --mask-ratio "$SPARK_MASK_RATIO" \
  --workers "$WORKERS" \
  --img-size "$PRETRAIN_IMG" \
  --patience "$SPARK_PATIENCE" \
  --min-delta "$SPARK_MIN_DELTA" \
  --amp

echo ""
echo "SparK pretraining completed"

SPARK_WEIGHTS="${SPARK_OUTPUT}/best_spark_model.pth"

if [ ! -f "$SPARK_WEIGHTS" ]; then
  echo "ERROR: best_spark_model.pth not found in $SPARK_OUTPUT"
  echo "Ensure spark_pretrain.py saves best_spark_model.pth (val attiva ⇒ early stopping)."
  exit 1
fi

echo "Using pretrained backbone: $SPARK_WEIGHTS"

# ==================================================
# Stage 2: PK-YOLO Training with Pretrained Backbone
# ==================================================
echo ""
echo "[Stage 2/2] Starting PK-YOLO Training..."
echo "Parameters:"
echo "  - Epochs:               $TRAIN_EPOCHS"
echo "  - Batch size:           $TRAIN_BATCH_SIZE"
echo "  - LR:                   $TRAIN_LR"
echo "  - Backbone LR mult:     $BACKBONE_LR_MULT"
echo "  - Img size:             $IMG_SIZE"
echo "  - Workers:              $WORKERS"
echo "  - Pretrained backbone:  $SPARK_WEIGHTS"
echo ""

python3 training_script.py \
  --data_dir "$DATA_DIR" \
  --output_dir "$TRAIN_OUTPUT" \
  --batch_size "$TRAIN_BATCH_SIZE" \
  --epochs "$TRAIN_EPOCHS" \
  --lr "$TRAIN_LR" \
  --img_size "$IMG_SIZE" \
  --workers "$WORKERS" \
  --spark_backbone_path "$SPARK_WEIGHTS" \
  --backbone_lr_mult "$BACKBONE_LR_MULT" \
  --mixed_precision

echo ""
echo "PK-YOLO training completed"

echo ""
echo "========================================================================"
echo "PIPELINE COMPLETED SUCCESSFULLY"
echo "========================================================================"
echo "Results:"
echo "  - SparK backbone:              $SPARK_WEIGHTS"
echo "  - PK-YOLO checkpoints:         $TRAIN_OUTPUT/checkpoints/"
echo "  - Best model:                   $TRAIN_OUTPUT/checkpoints/best_model.pth"
echo "  - Training visualizations:      $TRAIN_OUTPUT/training_visualizations/"
echo "  - Configuration:                $TRAIN_OUTPUT/config.yaml"
echo "========================================================================"
