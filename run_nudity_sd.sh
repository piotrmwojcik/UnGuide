#!/bin/bash
# Full training + generation: Nudity SD
# Steps: 1000 training, then generate images from I2P prompts
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

CONFIG="configs/nudity/nudity_sd.yaml"
OUTPUT_DIR="output/nudity_sd"
LORA_PATH="$OUTPUT_DIR/LoRA_fusion_model/hyper_lora_final.pth"

# Check wandb
WANDB_FLAG=""
python -c "import wandb; wandb.login()" 2>/dev/null && WANDB_FLAG="--use-wandb"
[ -n "$WANDB_FLAG" ] && echo "wandb: enabled" || echo "wandb: disabled"

echo "========================================"
echo "Nudity SD — Full Training"
echo "========================================"

# Train
python train.py --config "$CONFIG" $WANDB_FLAG

# Generate from I2P prompts
echo ""
echo "========================================"
echo "Generating images..."
echo "========================================"

python generate.py \
    --config "$CONFIG" \
    --task nudity \
    --lora-path "$LORA_PATH" \
    --prompts-csv data/I2P_prompts_4703.csv \
    --output-dir "$OUTPUT_DIR/images"

echo ""
echo "Done! Results in $OUTPUT_DIR/"
