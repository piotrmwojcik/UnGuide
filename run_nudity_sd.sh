#!/bin/bash
# Full training + generation: Nudity SD
# Supports GPU selection and optional train/generate modes
set -e

########################################
# ----------- ARG PARSING ------------
########################################

GPU_ID=0
MODE="full"   # full | train | generate

while [[ $# -gt 0 ]]; do
  case $1 in
    --gpu)
      GPU_ID="$2"
      shift 2
      ;;
    --train-only)
      MODE="train"
      shift
      ;;
    --generate-only)
      MODE="generate"
      shift
      ;;
    *)
      echo "Unknown argument: $1"
      exit 1
      ;;
  esac
done

########################################
# -------- GPU SELECTION --------------
########################################

echo "Using GPU: $GPU_ID"

########################################
# -------- PATHS ----------------------
########################################

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

CONFIG="configs/nudity/nudity_sd.yaml"
OUTPUT_DIR="output/nudity_sd"
LORA_PATH="$OUTPUT_DIR/LoRA_fusion_model/hyper_lora_final.pth"

########################################
# -------- WANDB CHECK ----------------
########################################

WANDB_FLAG=""
python -c "import wandb; wandb.login()" 2>/dev/null && WANDB_FLAG="--use-wandb"
[ -n "$WANDB_FLAG" ] && echo "wandb: enabled" || echo "wandb: disabled"

echo "========================================"
echo "Nudity SD — Mode: $MODE"
echo "========================================"

########################################
# -------- TRAIN ----------------------
########################################

if [[ "$MODE" == "full" || "$MODE" == "train" ]]; then
    echo "Starting training..."
    CUDA_VISIBLE_DEVICES=$GPU_ID accelerate launch --num_processes 1 --num_machines 1 \
        train.py --config "$CONFIG" $WANDB_FLAG
fi

########################################
# -------- GENERATE -------------------
########################################

if [[ "$MODE" == "full" || "$MODE" == "generate" ]]; then
    echo ""
    echo "========================================"
    echo "Generating images..."
    echo "========================================"

    CUDA_VISIBLE_DEVICES=$GPU_ID python generate.py \
        --config "$CONFIG" \
        --task nudity \
        --lora-path "$LORA_PATH" \
        --prompts-csv data/I2P_prompts_4703.csv \
        --output-dir "$OUTPUT_DIR/images"
fi

echo ""
echo "Done! Results in $OUTPUT_DIR/"
