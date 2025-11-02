#!/bin/bash
# PK-YOLO Training Pipeline: SparK Pretraining + Full Training

set -e 

# Data paths
DATA_DIR="./data"
OUTPUT_DIR="./experiments"

# Parameters (PK-YOLO paper Section 4.2)
IMG_SIZE=640
WORKERS=4

# SparK Pretraining Parameters 
SPARK_EPOCHS=300
SPARK_BATCH_SIZE=16
SPARK_LR=0.001
SPARK_MASK_RATIO=0.6
SPARK_PATCH_SIZE=16

# PK-YOLO Training Parameters (PK-YOLO paper Section 4.2)
TRAIN_EPOCHS=300
TRAIN_BATCH_SIZE=32
TRAIN_LR=0.0001
BACKBONE_LR_MULT=0.1

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
SPARK_WEIGHTS="experiments/spark_pretrain_20251012_182829/best_spark_model.pth"
TRAIN_OUTPUT="${OUTPUT_DIR}/pkyolo_train_${TIMESTAMP}"

mkdir -p "$TRAIN_OUTPUT"

echo "========================================================================"
echo "PK-YOLO Training Pipeline"
echo "========================================================================"
echo "Data directory: $DATA_DIR"
echo "SparK output: $SPARK_WEIGHTS"
echo "Training output: $TRAIN_OUTPUT"
echo "========================================================================"

echo "Using pretrained backbone: $SPARK_WEIGHTS"

# ==================================================
# Stage 2: PK-YOLO Training with Pretrained Backbone
# ==================================================

echo ""
echo "[Stage 2/2] Starting PK-YOLO Training..."
echo "Parameters:"
echo "  - Epochs: $TRAIN_EPOCHS"
echo "  - Batch size: $TRAIN_BATCH_SIZE"
echo "  - Learning rate: $TRAIN_LR"
echo "  - Backbone LR multiplier: $BACKBONE_LR_MULT"
echo "  - Using pretrained backbone: $SPARK_WEIGHTS"
echo ""

python3 training_script.py \
    --data_dir "$DATA_DIR" \
    --output_dir "$TRAIN_OUTPUT" \
    --batch_size $TRAIN_BATCH_SIZE \
    --epochs $TRAIN_EPOCHS \
    --lr $TRAIN_LR \
    --img_size $IMG_SIZE \
    --workers $WORKERS \
    --spark_backbone_path "$SPARK_WEIGHTS" \
    --backbone_lr_mult $BACKBONE_LR_MULT \
    --mixed_precision

echo ""
echo "PK-YOLO training completed"

echo ""
echo "========================================================================"
echo "PIPELINE COMPLETED SUCCESSFULLY"
echo "========================================================================"
echo "Results:"
echo "  - SparK backbone: $SPARK_WEIGHTS"
echo "  - PK-YOLO checkpoints: $TRAIN_OUTPUT/checkpoints/"
echo "  - Best model: $TRAIN_OUTPUT/checkpoints/best_model.pth"
echo "  - Training visualizations: $TRAIN_OUTPUT/training_visualizations/"
echo "  - Configuration: $TRAIN_OUTPUT/config.yaml"
echo "========================================================================"
