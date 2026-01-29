# UnGuide

Concept unlearning for diffusion models using HyperLoRA.

## Installation

```bash
pip install -r requirements.txt
```

**Note:** For NV-Embed embeddings, Hugging Face authentication is required:
```bash
huggingface-cli login
# Or set HF_TOKEN environment variable
export HF_TOKEN=your_token_here
```

## Training

### Stable Diffusion

```bash
# Celebrity unlearning (CLIP embeddings)
python train_sd.py --config configs/celebrity/train_celebrity_clip.yaml

# Celebrity unlearning (NV-Embed embeddings - requires HF auth)
python train_sd.py --config configs/celebrity/train_celebrity_nvembed.yaml

# Nudity unlearning
python train_sd.py --config configs/nudity/nudity_sd.yaml

# CIFAR-10 object unlearning
python train_sd.py --config configs/cifar_10/train_airplane.yaml
```

### Flux

```bash
python train_flux.py --config configs/nudity/nudity_flux.yaml
```

## Image Generation

Use `generate_sd.py` for Stable Diffusion models:

### Celebrity Task

Generate images for celebrity unlearning evaluation:

```bash
python generate_sd.py --task celebrity \
    --config configs/celebrity/train_celebrity_clip.yaml \
    --lora-path output/celebrity_clip/LoRA_fusion_model/hyper_lora.pth \
    --prompts-csv data/celebrity_eval.csv \
    --output-dir output/images
```

Or generate from config concepts (no CSV needed):

```bash
python generate_sd.py --task celebrity \
    --config configs/celebrity/train_celebrity_clip.yaml \
    --lora-path output/celebrity_clip/LoRA_fusion_model/hyper_lora.pth \
    --output-dir output/images
```

### Nudity Task

Generate images for nudity/NSFW unlearning evaluation:

```bash
python generate_sd.py --task nudity \
    --config configs/nudity/nudity_sd.yaml \
    --lora-path output/nudity_sd/LoRA_fusion_model/hyper_lora.pth \
    --prompts-csv data/I2P_prompts_4703.csv \
    --output-dir output/images
```

With nudity filtering (keeps only prompts with `nudity_percentage > 0`, sorted by nudity descending):

```bash
python generate_sd.py --task nudity \
    --config configs/nudity/nudity_sd.yaml \
    --lora-path output/nudity_sd/LoRA_fusion_model/hyper_lora.pth \
    --prompts-csv data/I2P_prompts_4703.csv \
    --output-dir output/images \
    --filter-nudity
```

### CIFAR-10 Task

Generate images for object unlearning evaluation:

```bash
python generate_sd.py --task cifar10 \
    --config configs/cifar_10/train_airplane.yaml \
    --lora-path output/cifar10_airplane/LoRA_fusion_model/hyper_lora.pth \
    --output-dir output/images \
    --samples-per-prompt 50
```

### Generation Options

| Argument | Description | Default |
|----------|-------------|---------|
| `--task` | Task type: `celebrity`, `nudity`, `cifar10` | Required |
| `--config` | Path to training config YAML | Required |
| `--lora-path` | Path to trained HyperLoRA weights | Required |
| `--output-dir` | Directory to save generated images | Required |
| `--prompts-csv` | CSV file with prompts (columns: `prompt`, `evaluation_seed`) | Optional |
| `--prompts-json` | JSON file with prompts (for cifar10) | Optional |
| `--n-images` | Limit number of images to generate | All |
| `--samples-per-prompt` | Images per prompt (cifar10) | 1 |
| `--steps` | DDIM sampling steps | 50 |
| `--guidance-scale` | CFG guidance scale | 7.5 |
| `--image-size` | Image size | 512 |
| `--filter-nudity` | Filter by `nudity_percentage > 0` | False |
| `--device` | Device to run on | `cuda:0` |

### Multi-GPU Generation

```bash
torchrun --nproc_per_node=4 generate_sd.py --task nudity \
    --config configs/nudity/nudity_sd.yaml \
    --lora-path output/nudity_sd/LoRA_fusion_model/hyper_lora.pth \
    --prompts-csv data/I2P_prompts_4703.csv \
    --output-dir output/images
```

### Flux Generation

For Flux models, use `generate_flux.py`:

```bash
python generate_flux.py \
    --csv_path data/I2P_prompts_4703.csv \
    --lora_path output/nudity_flux/LoRA_model/hyper_lora.pth \
    --output_dir output/flux_images
```

## Testing

Run a quick test of all training and generation pipelines:

```bash
./run_test.sh
```

This runs 5 short trainings (10 steps each) and generates one image from each model.

## Evaluation

Evaluation scripts are in the `eval/` directory:

```bash
python eval/compute_fid.py --generated output/images --reference data/reference
python eval/compute_clip_score.py --images output/images --prompts data/prompts.csv
```

## Config Structure

Training configs are YAML files with the following structure:

```yaml
UnGuide_Example:
  # Target concepts to remove
  concepts:
    - "concept1"
    - "concept2"

  # What removed concepts should map to
  mapping_concept:
    - "replacement1"
    - "replacement2"

  # Retain prompts CSV
  retain_csv_path: ./data/retain.csv

  # Embedding model: clip, clip_huge, or nv_embed
  # Note: nv_embed requires HF authentication
  embedding_model: clip
  use_pooler: true

  # HyperLoRA parameters
  rank: 6
  lora_alpha: 0.01
  internal_size: 512
  hyper_train_steps: 300

  # Training parameters
  learning_rate: 1.0e-03
  max_train_steps: 1000

  # Paths
  output_dir: output/example
  model_config: ./configs/stable-diffusion/v1-inference.yaml
  pretrained_model_name_or_path: models/sd-v1-4.ckpt
```

## Project Structure

```
UnGuide/
├── train_sd.py              # Stable Diffusion training
├── train_flux.py            # Flux training
├── generate_sd.py           # SD image generation
├── generate_flux.py         # Flux image generation
├── hyper_lora.py            # HyperLoRA implementation
├── run_test.sh              # Test script
├── configs/
│   ├── celebrity/
│   │   ├── train_celebrity_clip.yaml
│   │   └── train_celebrity_nvembed.yaml
│   ├── nudity/
│   │   ├── nudity_sd.yaml
│   │   └── nudity_flux.yaml
│   ├── cifar_10/
│   │   └── train_airplane.yaml (and other classes)
│   └── stable-diffusion/
│       └── v1-inference.yaml
├── configs_test/            # Test configs (10 steps)
├── data/
│   ├── I2P_prompts_4703.csv
│   ├── coco_30k.csv
│   ├── cifar100.csv
│   └── celebrity_retain_improved.csv
├── eval/                    # Evaluation scripts
├── utils/                   # Utility modules
└── ldm/                     # Latent diffusion model code
```
