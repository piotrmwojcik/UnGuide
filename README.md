# UnGuide: Learning to Forget with LoRA-Guided Diffusion Models

![teaser](assets/teaser.jpg)

> Recent advances in large-scale text-to-image diffusion models have heightened concerns about their potential misuse, especially in generating harmful or misleading content. This underscores the urgent need for effective machine unlearning, i.e., removing specific knowledge or concepts from pretrained models without compromising overall performance. One possible approach is Low-Rank Adaptation (LoRA), which offers an efficient means to fine-tune models for targeted unlearning. However, LoRA often inadvertently alters unrelated content, leading to diminished image fidelity and realism. To address this limitation, we introduce UnGuide—a novel approach which incorporates UnGuidance, a dynamic inference mechanism that leverages Classifier-Free Guidance (CFG) to
exert precise control over the unlearning process. UnGuide modulates the guidance scale based on the stability of a few first steps of denoising processes, enabling selective unlearning by LoRA adapter. For prompts containing the erased concept, the LoRA module predominates and is counter balanced by the base model; for unrelated prompts, the base model governs generation, preserving content fidelity. Empirical results demonstrate that UnGuide achieves controlled concept
removal and retains the expressive power of diffusion models, outperforming existing LoRA-based methods in both object erasure and explicit content removal tasks.



## 📦 Environment Setup
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## 🧠 Download Pretrained Model
Download the original Stable Diffusion model:
```bash
mkdir -p models
wget -O models/sd-v1-4.ckpt "https://huggingface.co/CompVis/stable-diffusion-v-1-4-original/resolve/main/sd-v1-4.ckpt?download=true"
```

## 🚀 Training (Concept Unlearning with LoRA)
```bash
python train.py \
  --ckpt_path models/sd-v1-4.ckpt \
  --prompts_json data/cat.json \
  --iterations 200 \
  --lr 3e-5 \
  --output_dir output
```
Customize options using python train.py --help

## 🧪 Latent Difference Analysis
Compare latent outputs between the original and LoRA-tuned models:

```bash
python compute_lora_diff.py \
  --config configs/stable-diffusion/v1-inference.yaml \
  --ckpt models/sd-v1-4.ckpt \
  --lora output/.../models/lora.pth \
  --prompts_json data/cat.json \
  --output_dir output/... \
  --device cuda:0 \
  --seed 42 \
  --ddim_steps 50 \
  --t_enc 40 \
  --repeats 30 \
  --n_samples 1
```

## 🖼️ Image Generation

```bash
python generate_images.py \
  --config configs/stable-diffusion/v1-inference.yaml \
  --ckpt models/sd-v1-4.ckpt \
  --output_dir output/ \
  --samples 10 \
  --steps 50 \
  --w1 -1 \
  --w2 2 \
  --seed 2024 \
  --device cuda:0
```

## NSFW
Run auto-guided generation only on prompts with nudity present (e.g., I2P dataset):

```bash
python generate_images_nsfw.py \
  --csv_path data/I2P_prompts_4703.csv \
  --output_dir output/.../ \
  --config configs/stable-diffusion/v1-inference.yaml \
  --ckpt models/sd-v1-4.ckpt \
  --alpha 8 \
  --image_size 512 \
  --ddim_steps 50 \
  --t_enc 40 \
  --batch_size 1 \
  --repeats 10 \
  --device cuda:0
```