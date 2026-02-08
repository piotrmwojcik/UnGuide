#!/bin/bash
# Full training + generation: Nudity Flux
# Steps: 6000 training, then generate images from I2P prompts
# NOTE: Flux requires ~45GB VRAM. Use a GPU with sufficient memory.
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

CONFIG="configs/nudity/nudity_flux.yaml"
OUTPUT_DIR="output/nudity_flux"
LORA_PATH="$OUTPUT_DIR/LoRA_model/hyper_lora_final.pth"

# Check wandb
WANDB_FLAG=""
python -c "import wandb; wandb.login()" 2>/dev/null && WANDB_FLAG="--use-wandb"
[ -n "$WANDB_FLAG" ] && echo "wandb: enabled" || echo "wandb: disabled"

echo "========================================"
echo "Nudity Flux — Full Training"
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
    --csv_path data/I2P_prompts_4703.csv \
    --lora_path "$LORA_PATH" \
    --output_dir "$OUTPUT_DIR/images"

echo ""
echo "Done! Results in $OUTPUT_DIR/"
