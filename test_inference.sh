#!/bin/bash

# Multimodal PK-YOLO Inference Testing Script
# This script runs inference testing on your trained model

echo "=========================================="
echo "Multimodal PK-YOLO Inference Testing"
echo "=========================================="

# Configuration
DATA_DIR="/data"  # Update this to your data directory path
MODEL_PATH="outputs/checkpoints/final_model.pth"  
OUTPUT_DIR="inference_results"
BATCH_SIZE=4
IMG_SIZE=640
CONF_THRESHOLD=0.25
NMS_THRESHOLD=0.45
DEVICE="cuda"  # or "cpu"

# Check if data directory exists
if [ ! -d "$DATA_DIR" ]; then
    echo "Error: Data directory $DATA_DIR does not exist!"
    echo "Please update DATA_DIR in this script to point to your data directory."
    exit 1
fi

# Check if test directory exists
if [ ! -d "$DATA_DIR/test" ]; then
    echo "Error: Test directory $DATA_DIR/test does not exist!"
    echo "Please ensure your data directory has the following structure:"
    echo "  $DATA_DIR/"
    echo "    test/"
    echo "      images/  (containing .png files)"
    echo "      labels/  (containing .txt files)"
    exit 1
fi

# Check if model exists
if [ ! -f "$MODEL_PATH" ]; then
    echo "Error: Model file $MODEL_PATH does not exist!"
    echo "Please update MODEL_PATH in this script to point to your trained model."
    echo "Common locations:"
    echo "  - outputs/checkpoints/best_model.pth"
    echo "  - outputs/checkpoints/final_model.pth"
    echo "  - outputs/checkpoints/checkpoint_epoch_XX.pth"
    exit 1
fi

echo "Configuration:"
echo "  Data directory: $DATA_DIR"
echo "  Model path: $MODEL_PATH"
echo "  Output directory: $OUTPUT_DIR"
echo "  Batch size: $BATCH_SIZE"
echo "  Image size: $IMG_SIZE"
echo "  Confidence threshold: $CONF_THRESHOLD"
echo "  NMS threshold: $NMS_THRESHOLD"
echo "  Device: $DEVICE"
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Run inference testing with full evaluation
echo "Running full inference testing..."
python test_inference.py \
    --data_dir "$DATA_DIR" \
    --model_path "$MODEL_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --batch_size $BATCH_SIZE \
    --img_size $IMG_SIZE \
    --conf_threshold $CONF_THRESHOLD \
    --nms_threshold $NMS_THRESHOLD \
    --device "$DEVICE" \
    --save_visualizations

echo ""
echo "Inference testing completed!"
echo "Results saved to: $OUTPUT_DIR"
echo ""
echo "Files generated:"
echo "  - $OUTPUT_DIR/inference_results.json  (detailed metrics)"
echo "  - $OUTPUT_DIR/visualizations/         (prediction visualizations)"
echo ""

# Quick test option (uncomment to run a quick test on limited samples)
# echo "Running quick test (limited samples)..."
# python test_inference.py \
#     --data_dir "$DATA_DIR" \
#     --model_path "$MODEL_PATH" \
#     --output_dir "${OUTPUT_DIR}_quick" \
#     --batch_size $