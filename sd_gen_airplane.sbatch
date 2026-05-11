#!/bin/bash
#SBATCH --job-name=airplane-gen
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --array=0-3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32GB
#SBATCH --time=01:60:00
#SBATCH --output=logs/airplane_gen_%A_%a.out
#SBATCH --error=logs/airplane_gen_%A_%a.err

set -euo pipefail

mkdir -p logs

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate unhype

export SCRATCH="/net/tscratch/people/plgpiotrwojcik"
export SCRATCH_HOME="$SCRATCH"

export HF_HOME="$SCRATCH/.cache/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export HF_HUB_DISABLE_XET=1

mkdir -p \
  "$HF_HOME" \
  "$HF_HUB_CACHE" \
  "$TRANSFORMERS_CACHE" \
  "$HF_DATASETS_CACHE" \
  "$SCRATCH/hf_models" \
  "$SCRATCH/generated/airplane"

cd /net/people/plgrid/plgpiotrwojcik/UnGuide

export CUDA_VISIBLE_DEVICES=0
export LOCAL_RANK=0
export RANK="$SLURM_ARRAY_TASK_ID"
export WORLD_SIZE=4

echo "SLURM_ARRAY_JOB_ID=$SLURM_ARRAY_JOB_ID"
echo "SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID"
echo "RANK=$RANK"
echo "WORLD_SIZE=$WORLD_SIZE"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
nvidia-smi || true

python -u -X faulthandler generate.py \
  --task cifar10 \
  --config configs/cifar_camera/train_airplane.yaml \
  --lora-path /net/tscratch/people/plgpiotrwojcik/output/cifar10_airplane/LoRA_fusion_model/hyper_lora_final.pth \
  --output-dir "$SCRATCH/generated/airplane" \
  --samples-per-prompt 200 \
  --steps 50 \
  --guidance-scale 7.5 \
  --ddim-eta 0.0 \
  --image-size 512 \
  --seed 2024 \
  --device cuda:0