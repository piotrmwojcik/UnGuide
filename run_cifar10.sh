#!/bin/bash
# Full training + generation: CIFAR-10 Airplane
# Steps: 1000 training, then generate images
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

CONFIG="configs/cifar_10/train_airplane.yaml"
OUTPUT_DIR="output/cifar10_airplane"
LORA_PATH="$OUTPUT_DIR/LoRA_fusion_model/hyper_lora_final.pth"

# Check wandb
WANDB_FLAG=""
python -c "import wandb; wandb.login()" 2>/dev/null && WANDB_FLAG="--use-wandb"
[ -n "$WANDB_FLAG" ] && echo "wandb: enabled" || echo "wandb: disabled"

echo "========================================"
echo "CIFAR-10 Airplane — Full Training"
echo "========================================"

# Train
python train.py --config "$CONFIG" $WANDB_FLAG

# Generate
echo ""
echo "========================================"
echo "Generating images..."
echo "========================================"

python generate.py \
    --config "$CONFIG" \
    --task cifar10 \
    --lora-path "$LORA_PATH" \
    --output-dir "$OUTPUT_DIR/images" \
    --samples-per-prompt 10

echo ""
echo "Done! Results in $OUTPUT_DIR/"
