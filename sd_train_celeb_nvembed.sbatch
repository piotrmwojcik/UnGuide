#!/bin/bash
#SBATCH --job-name=unhype-nvm
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --nodes=1
#SBATCH -A plgunhype-gpu-a100
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32GB
#SBATCH --time=10:00:00
#SBATCH --output=logs/nvembed_%j.out
#SBATCH --error=logs/nvembed_%j.err

export HF_HOME=/net/tscratch/people/plgpiotrwojcik/.cache/huggingface
export HUGGINGFACE_HUB_CACHE=/net/tscratch/people/plgpiotrwojcik/.cache/huggingface/hub
#export HF_TOKEN=<>
export WANDB_API_KEY="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"

source ~/.bashrc
conda activate unhype

export SCRATCH_HOME=/net/tscratch/people/plgpiotrwojcik/


cd /net/tscratch/people/plgpiotrwojcik/projects/UnGuide/UnHype

CUDA_VISIBLE_DEVICES=0 python train.py \
    --config configs/celebrity/train_celebrity_nvembed.yaml \
    --use-wandb