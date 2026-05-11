#!/bin/bash
#SBATCH --job-name=un-celebs
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32GB
#SBATCH --time=24:00:00
#SBATCH --output=logs/job_%j.out
#SBATCH --error=logs/job_%j.err


source ~/.bashrc
conda activate unhype

export SCRATCH_HOME=/net/tscratch/people/plgpiotrwojcik/

set -euo pipefail

cd /net/people/plgrid/plgpiotrwojcik/UnGuide
mkdir -p logs

# CHANGE this to the actual python inside the env:
PY=/net/tscratch/people/plgpiotrwojcik/conda-envs/unhype/bin/python

# optional sanity checks
$PY -V
$PY -c "import torch; print('torch', torch.__version__)"

# run generation (your bash script will call python/generate.py internally,
# so make sure it uses the same interpreter; simplest is to call generate.py directly)
#$PY generate.py \
#  --config configs/nudity/nudity_sd.yaml \
#  --prompts-csv data/I2P_prompts_4703.csv \
#  --lora-path output/nudity_sd/LoRA_fusion_model/hyper_lora_final.pth \
#  --task nudity \
#  --output-dir output/nudity_sd/images

export HF_HOME=/net/tscratch/people/plgpiotrwojcik/.cache/huggingface
export HUGGINGFACE_HUB_CACHE=/net/tscratch/people/plgpiotrwojcik/.cache/huggingface/hub


$PY generate.py \
    --config configs/celebrity/train_celebrity_nvembed.yaml \
    --task celebrity \
    --lora-path /net/tscratch/people/plgpiotrwojcik/output/celebrity_nvembed/LoRA_fusion_model/hyper_lora_1999.pth \
    --prompts-csv prompts/celebrity_100_concepts.csv \
    --output-dir $SCRATCH_HOME/output/celebrity_100_camera_nvembed/images
