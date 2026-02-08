#!/bin/bash
# Full training + generation: Celebrity (CLIP embeddings)
# Steps: 6000 training, then generate images
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

CONFIG="configs/celebrity/train_celebrity_clip.yaml"
OUTPUT_DIR="output/celebrity_clip"
LORA_PATH="$OUTPUT_DIR/LoRA_fusion_model/hyper_lora_final.pth"

# Check wandb
WANDB_FLAG=""
python -c "import wandb; wandb.login()" 2>/dev/null && WANDB_FLAG="--use-wandb"
[ -n "$WANDB_FLAG" ] && echo "wandb: enabled" || echo "wandb: disabled"

echo "========================================"
echo "Celebrity CLIP — Full Training"
echo "========================================"

# Train
python train.py --config "$CONFIG" $WANDB_FLAG

# Generate — use celebrity_retain CSV for eval prompts
echo ""
echo "========================================"
echo "Generating images..."
echo "========================================"

python generate.py \
    --config "$CONFIG" \
    --task celebrity \
    --lora-path "$LORA_PATH" \
    --prompts-csv data/celebrity_retain_improved.csv \
    --output-dir "$OUTPUT_DIR/images" \
    --n-images 200

echo ""
echo "Done! Results in $OUTPUT_DIR/"
