#!/usr/bin/env python3
"""
Unified image generation script for HyperLoRA models.
Supports celebrity, nudity, and cifar10 tasks.

Usage:
    python generate_images.py --task celebrity --config configs/celebrity/train_celebrity_100_final.yaml \
        --lora-path output/LoRA_fusion_model/hyper_lora.pth --prompts-csv data/celebrity_eval.csv --output-dir output/images

    python generate_images.py --task nudity --config configs/nudity/nudity_10.yaml \
        --lora-path output/LoRA_fusion_model/hyper_lora.pth --prompts-csv data/I2P_prompts_4703.csv --output-dir output/images

    python generate_images.py --task cifar10 --config configs/cifar_10/train_airplane.yaml \
        --lora-path output/LoRA_fusion_model/hyper_lora.pth --output-dir output/images
"""

import argparse
import os
import re
import time
import json
import torch
import numpy as np
import pandas as pd
import yaml
from functools import partial
from tqdm import tqdm
from torchvision.transforms.functional import to_pil_image

from transformers import CLIPTextModel, CLIPTokenizer
from hyper_lora import HyperLoRALinear, HypernetworkManager, inject_hyper_lora
from ldm.models.diffusion.ddimcopy import DDIMSampler
from utils import load_model_from_config, set_seed

# Lazy imports for NV-Embed
_nv_embed_module = None

def _load_nv_embed_module():
    global _nv_embed_module
    if _nv_embed_module is None:
        import utils.nv_embed_utils as nv_embed_utils
        _nv_embed_module = nv_embed_utils
    return _nv_embed_module


def parse_args():
    parser = argparse.ArgumentParser(
        description='Unified image generation with HyperLoRA'
    )
    # Required arguments
    parser.add_argument('--task', type=str, required=True,
                        choices=['celebrity', 'nudity', 'cifar10'],
                        help='Task type: celebrity, nudity, or cifar10')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to training config YAML (contains embedding model, LoRA params)')
    parser.add_argument('--lora-path', type=str, required=True,
                        help='Path to trained HyperLoRA weights (.pth file or directory)')
    parser.add_argument('--output-dir', type=str, required=True,
                        help='Directory to save generated images')

    # Optional prompt source override
    parser.add_argument('--prompts-csv', type=str, default=None,
                        help='CSV file with prompts (columns: prompt, evaluation_seed)')
    parser.add_argument('--prompts-json', type=str, default=None,
                        help='JSON file with prompts (for cifar10: target, synonyms, other)')

    # Generation parameters
    parser.add_argument('--n-images', type=int, default=None,
                        help='Limit number of images to generate')
    parser.add_argument('--samples-per-prompt', type=int, default=1,
                        help='Number of images per prompt (for cifar10)')
    parser.add_argument('--steps', type=int, default=50,
                        help='Number of DDIM sampling steps')
    parser.add_argument('--guidance-scale', type=float, default=7.5,
                        help='CFG guidance scale')
    parser.add_argument('--ddim-eta', type=float, default=0.0,
                        help='DDIM eta parameter')
    parser.add_argument('--image-size', type=int, default=512,
                        help='Image size for generation')

    # Filtering
    parser.add_argument('--filter-nudity', action='store_true',
                        help='Filter CSV to keep only rows with nudity_percentage > 0 and sort by nudity descending')

    # Model paths (can override config)
    parser.add_argument('--model-config', type=str, default=None,
                        help='Override: path to SD model config YAML')
    parser.add_argument('--ckpt', type=str, default=None,
                        help='Override: path to SD model checkpoint')

    # Runtime
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='Device to run generation on')
    parser.add_argument('--seed', type=int, default=2024,
                        help='Random seed for initialization')

    return parser.parse_args()


def load_config(config_path: str) -> dict:
    """Load and flatten YAML config."""
    with open(config_path, 'r') as f:
        raw = yaml.safe_load(f)
    # Config may be nested under a key like 'UnGuide_Celebrity_100'
    if len(raw) == 1:
        key = list(raw.keys())[0]
        return raw[key]
    return raw


def coerce_prompt(v):
    """Normalize prompt value from CSV."""
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return ""
    if isinstance(v, (list, tuple, set)):
        return ", ".join(str(x).strip() for x in v if str(x).strip())
    s = str(v).strip()
    s = re.sub(r'^\s*prompt\s*[:\-]?\s*', "", s, flags=re.I)
    m = re.match(r'^\[\s*(.*)\s*\]$', s)
    if m:
        parts = [p.strip() for p in m.group(1).split(",")]
        return ", ".join(p for p in parts if p)
    return s


def load_lora_weights(model, lora_path: str, device: torch.device):
    """Load HyperLoRA weights into model."""
    print(f"Loading LoRA weights from: {lora_path}")

    if not os.path.exists(lora_path):
        raise FileNotFoundError(f"LoRA checkpoint not found: {lora_path}")

    lora_state_dict = torch.load(lora_path, map_location=device)
    print(f"Found {len(lora_state_dict)} parameters in checkpoint")

    diffusion_model = model.model.diffusion_model
    sd = diffusion_model.state_dict()

    updated = 0
    skipped = []

    with torch.no_grad():
        for k, v in lora_state_dict.items():
            if k in sd:
                if torch.is_tensor(sd[k]) and torch.is_tensor(v) and sd[k].shape == v.shape:
                    sd[k].copy_(v.to(sd[k].dtype).to(device))
                    updated += 1
                else:
                    skipped.append((k, f"shape mismatch"))
            else:
                skipped.append((k, "no such key"))

    print(f"[LoRA] Loaded {updated} tensors, skipped {len(skipped)}")
    return model


def setup_embedding_model(embedding_model: str, use_pooler: bool, device: torch.device):
    """
    Setup embedding model and return (embed_fn, clip_size).

    embed_fn: function that takes a prompt string and returns embedding tensor
    """
    if embedding_model == 'nv_embed':
        nv_mod = _load_nv_embed_module()
        nv_model, nv_tokenizer = nv_mod.load_nv_embed_model(device=device, dtype=torch.float16)
        clip_size = nv_mod.NV_EMBED_DIM  # 4096

        def embed_fn(prompt):
            return nv_mod.compute_nv_embed([prompt], nv_model, nv_tokenizer, device, batch_size=1)

        return embed_fn, clip_size

    elif embedding_model == 'clip_huge':
        tokenizer = CLIPTokenizer.from_pretrained("laion/CLIP-ViT-H-14-laion2B-s32B-b79K")
        clip_model = CLIPTextModel.from_pretrained("laion/CLIP-ViT-H-14-laion2B-s32B-b79K").to(device).eval()
        clip_size = 1024  # ViT-H uses 1024

        def embed_fn(prompt):
            inputs = tokenizer(
                prompt,
                max_length=tokenizer.model_max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            ).to(device).input_ids
            with torch.no_grad():
                return clip_model(inputs).pooler_output.detach()

        return embed_fn, clip_size

    else:  # 'clip' (default)
        tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
        clip_model = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14").to(device).eval()
        clip_size = 768 if use_pooler else 512

        def embed_fn(prompt):
            inputs = tokenizer(
                prompt,
                max_length=tokenizer.model_max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            ).to(device).input_ids
            with torch.no_grad():
                if use_pooler:
                    return clip_model(inputs).pooler_output.detach()
                else:
                    return clip_model(inputs).last_hidden_state.detach()

        return embed_fn, clip_size


class CombinedCFGModelSingleModel:
    """Wrapper that toggles LoRA for conditional vs unconditional passes."""

    def __init__(self, model):
        self.model = model
        self.device = model.device

    def apply_model(self, x, t, c):
        b2 = x.shape[0]
        assert b2 % 2 == 0
        b = b2 // 2

        x_uncond, x_cond = x[:b], x[b:]
        t_uncond, t_cond = t[:b], t[b:]

        if isinstance(c, dict):
            c_uncond = {k: [v[:b] for v in c[k]] if isinstance(c[k], list) else c[k][:b] for k in c}
            c_cond = {k: [v[b:] for v in c[k]] if isinstance(c[k], list) else c[k][b:] for k in c}
        else:
            c_uncond, c_cond = c[:b], c[b:]

        # Unconditional: LoRA OFF
        with self.model.hyper.no_lora():
            out_uncond = self.model.apply_model(x_uncond, t_uncond, c_uncond)
        # Conditional: LoRA ON
        out_cond = self.model.apply_model(x_cond, t_cond, c_cond)

        return torch.cat([out_uncond, out_cond], dim=0)

    def get_learned_conditioning(self, prompts):
        return self.model.get_learned_conditioning(prompts)

    def decode_first_stage(self, z):
        return self.model.decode_first_stage(z)

    def eval(self):
        self.model.eval()
        return self

    def __getattr__(self, name):
        return getattr(self.model, name)


def generate_image(sampler, model, prompt: str, device: torch.device,
                   steps: int = 50, eta: float = 0.0, start_code=None,
                   guidance_scale: float = 7.5):
    """Generate a single image."""
    if start_code is None:
        start_code = torch.randn(1, 4, 64, 64, device=device)

    model.eval()
    with torch.no_grad(), torch.autocast(device_type=device.type, enabled=(device.type == "cuda")):
        cond = model.get_learned_conditioning([prompt])
        uncond = model.get_learned_conditioning([""])

        samples, _ = sampler.sample(
            S=steps,
            conditioning={"c_crossattn": [cond]},
            batch_size=1,
            shape=start_code.shape[1:],
            verbose=False,
            unconditional_guidance_scale=guidance_scale,
            unconditional_conditioning={"c_crossattn": [uncond]},
            eta=eta,
            x_T=start_code,
        )
        decoded = model.decode_first_stage(samples)
        decoded = (decoded + 1.0) / 2.0
        decoded = torch.clamp(decoded, 0.0, 1.0)
        return decoded


def build_prompts_celebrity(config: dict, csv_path: str = None):
    """
    Build prompts for celebrity task.
    Returns list of (prompt, seed, filename) tuples.
    """
    if csv_path:
        df = pd.read_csv(csv_path)
        if 'prompt' not in df.columns or 'evaluation_seed' not in df.columns:
            raise ValueError("Celebrity CSV must have 'prompt' and 'evaluation_seed' columns")
        prompts = []
        for idx, row in df.iterrows():
            prompt = coerce_prompt(row['prompt'])
            seed = int(row['evaluation_seed'])
            # Sanitize prompt for filename
            safe_prompt = re.sub(r'[^\w\s-]', '', prompt)[:50].strip()
            filename = f"{safe_prompt}_{seed}.png"
            prompts.append((prompt, seed, filename))
        return prompts
    else:
        # Generate from config concepts
        concepts = config.get('concepts', [])
        prompts = []
        base_seed = config.get('seed', 2024)
        for i, concept in enumerate(concepts):
            prompt = f"a photo of {concept}"
            seed = base_seed + i
            safe_name = re.sub(r'[^\w\s-]', '', concept)[:30].strip().replace(' ', '_')
            filename = f"{safe_name}_{seed}.png"
            prompts.append((prompt, seed, filename))
        return prompts


def build_prompts_nudity(config: dict, csv_path: str, filter_nudity: bool = False):
    """
    Build prompts for nudity task from CSV.
    Returns list of (prompt, seed, filename) tuples.
    """
    if not csv_path:
        raise ValueError("Nudity task requires --prompts-csv")

    df = pd.read_csv(csv_path, index_col=0)

    if filter_nudity and "nudity_percentage" in df.columns:
        df["nudity_percentage"] = pd.to_numeric(df["nudity_percentage"], errors="coerce")
        df = df[df["nudity_percentage"].gt(0)]
        df = df.sort_values(by="nudity_percentage", ascending=False)
        print(f"[Filter] Kept {len(df)} rows with nudity_percentage > 0")

    prompts = []
    for idx, row in df.iterrows():
        prompt = coerce_prompt(row.get("prompt", ""))
        if not prompt:
            continue
        seed = int(row.get("evaluation_seed", 0))
        filename = f"{idx:05d}.png"
        prompts.append((prompt, seed, filename))

    return prompts


def build_prompts_cifar10(config: dict, json_path: str = None, samples_per_prompt: int = 1):
    """
    Build prompts for cifar10 task.
    Returns list of (prompt, seed, filename, subdir) tuples.
    """
    if json_path:
        with open(json_path, 'r') as f:
            data = json.load(f)
        all_prompts = [data.get("target")] + data.get("synonyms", []) + data.get("other", [])
        all_prompts = [p for p in all_prompts if p]
    else:
        # Use concepts from config
        concepts = config.get('concepts', [])
        all_prompts = [f"a photo of {c}" for c in concepts]
        # Also add diagnostic prompts if available
        diag = config.get('diagnostic_prompts', [])
        all_prompts.extend(diag)

    base_seed = config.get('seed', 2024)
    prompts = []

    for prompt in all_prompts:
        # Extract class name for subdirectory
        class_name = prompt.split()[-1] if prompt else "unknown"
        class_name = re.sub(r'[^\w-]', '', class_name)

        for i in range(samples_per_prompt):
            seed = base_seed + i
            filename = f"{i:05d}.png"
            prompts.append((prompt, seed, filename, class_name))

    return prompts


def main():
    # Multi-GPU setup
    LOCAL_RANK = int(os.environ.get("LOCAL_RANK", "0"))
    RANK = int(os.environ.get("RANK", 0))
    WORLD_SIZE = int(os.environ.get("WORLD_SIZE", 1))

    if torch.cuda.is_available():
        torch.cuda.set_device(LOCAL_RANK)

    args = parse_args()

    # Load config
    config = load_config(args.config)
    print(f"Loaded config from {args.config}")

    # Determine device
    if args.device.startswith('cuda') and torch.cuda.is_available():
        device = torch.device(f"cuda:{LOCAL_RANK}")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device} (RANK {RANK}/{WORLD_SIZE})")

    # Set seed for initialization
    set_seed(args.seed)

    # Get paths from config or args
    model_config = args.model_config or config.get('model_config', './configs/stable-diffusion/v1-inference.yaml')
    ckpt_path = args.ckpt or config.get('pretrained_model_name_or_path', 'models/sd-v1-4.ckpt')

    # Get HyperLoRA params from config
    embedding_model = config.get('embedding_model', 'clip')
    use_pooler = config.get('use_pooler', True)
    rank = config.get('rank', 1)
    lora_alpha = config.get('lora_alpha', 1.0)
    hidden_size = config.get('hidden_size', 512)
    hyper_train_steps = config.get('hyper_train_steps', 300)
    use_orig_concat = config.get('use_orig_concat', False)

    print(f"\nHyperLoRA config:")
    print(f"  embedding_model: {embedding_model}")
    print(f"  rank: {rank}, alpha: {lora_alpha}, hidden_size: {hidden_size}")
    print(f"  hyper_train_steps: {hyper_train_steps}, use_orig_concat: {use_orig_concat}")

    # Setup embedding model
    embed_fn, clip_size = setup_embedding_model(embedding_model, use_pooler, device)
    print(f"  clip_size: {clip_size}")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Build prompts based on task
    print(f"\nTask: {args.task}")
    if args.task == 'celebrity':
        prompt_data = build_prompts_celebrity(config, args.prompts_csv)
        use_subdirs = False
    elif args.task == 'nudity':
        prompt_data = build_prompts_nudity(config, args.prompts_csv, args.filter_nudity)
        use_subdirs = False
    else:  # cifar10
        prompt_data = build_prompts_cifar10(config, args.prompts_json, args.samples_per_prompt)
        use_subdirs = True

    print(f"Found {len(prompt_data)} prompts to generate")

    # Apply n_images limit
    if args.n_images is not None:
        prompt_data = prompt_data[:args.n_images]
        print(f"Limited to {len(prompt_data)} images")

    # Load SD model
    print(f"\nLoading Stable Diffusion from {ckpt_path}...")
    model = load_model_from_config(model_config, ckpt_path, device)

    # Find LoRA checkpoint file
    lora_file = args.lora_path
    if os.path.isdir(args.lora_path):
        candidates = [f for f in os.listdir(args.lora_path) if f.startswith('hyper_lora') and f.endswith('.pth')]
        if candidates:
            candidates.sort()
            lora_file = os.path.join(args.lora_path, candidates[-1])
        else:
            lora_file = os.path.join(args.lora_path, "hyper_lora.pth")

    # Setup HyperLoRA
    print(f"\nSetting up HyperLoRA (rank={rank}, clip_size={clip_size})...")
    hyper_lora_factory = partial(
        HyperLoRALinear,
        clip_size=clip_size,
        rank=rank,
        alpha=lora_alpha,
        train_steps=hyper_train_steps,
        internal_size=hidden_size,
        use_orig_concat=use_orig_concat,
    )

    model.hyper = HypernetworkManager()
    hyper_lora_layers = inject_hyper_lora(
        model.model.diffusion_model, ["attn2.to_k", "attn2.to_v"], hyper_lora_factory
    )
    for layer_name, layer in hyper_lora_layers:
        layer.set_parent_model(model)
        model.hyper.add_hyperlora(layer_name, layer.hyper_lora)

    print(f"Injected HyperLoRA into {len(hyper_lora_layers)} layers")

    # Load LoRA weights
    load_lora_weights(model, lora_file, device)

    # Setup sampler with CFG wrapper
    combined_model = CombinedCFGModelSingleModel(model).eval()
    sampler = DDIMSampler(model=combined_model)

    # Generate images
    print(f"\nGenerating images...")
    images_generated = 0

    for i, item in enumerate(tqdm(prompt_data, desc="Generating")):
        # Handle different tuple formats
        if use_subdirs:
            prompt, seed, filename, subdir = item
            save_dir = os.path.join(args.output_dir, subdir)
            os.makedirs(save_dir, exist_ok=True)
        else:
            prompt, seed, filename = item
            save_dir = args.output_dir

        image_path = os.path.join(save_dir, filename)

        # Skip if already exists
        if os.path.exists(image_path):
            continue

        # Multi-GPU: skip if not our turn
        if i % WORLD_SIZE != RANK:
            continue

        start_time = time.time()

        # Set seed and generate start code
        set_seed(seed)
        gen = torch.Generator(device=device).manual_seed(seed)
        start_code = torch.randn(
            1, 4, args.image_size // 8, args.image_size // 8,
            generator=gen, device=device
        )

        # Compute embedding and set HyperLoRA context
        t_prompt = embed_fn(prompt)
        hyper_device = model.hyper.hyper_layers[0].alpha.device if model.hyper.hyper_layers else device
        weight_dtype = model.hyper.hyper_layers[0].alpha.dtype if model.hyper.hyper_layers else torch.float32

        t_prompt = t_prompt.to(dtype=weight_dtype, device=hyper_device)
        timestep = torch.tensor([hyper_train_steps], dtype=weight_dtype, device=hyper_device)

        model.hyper.set_context(t_prompt, timestep)
        model.hyper.compute_and_cache_loras(t_prompt, timestep)

        # Generate image
        with torch.no_grad():
            img = generate_image(
                sampler=sampler,
                model=combined_model,
                prompt=prompt,
                device=device,
                steps=args.steps,
                eta=args.ddim_eta,
                start_code=start_code,
                guidance_scale=args.guidance_scale,
            )

        # Save image
        img_np = img[0].cpu().permute(1, 2, 0).numpy()
        img_pil = to_pil_image((img_np * 255).astype(np.uint8))
        img_pil.save(image_path, format='PNG')

        images_generated += 1
        elapsed = time.time() - start_time

        if images_generated <= 3 or images_generated % 50 == 0:
            print(f"[{images_generated}] '{prompt[:40]}...' -> {filename} ({elapsed:.2f}s)")

    print(f"\nDone! Generated {images_generated} images in {args.output_dir}")


if __name__ == "__main__":
    main()
