#!/usr/bin/env python3
"""
Celebrity Image Generation with CFG and Multi-GPU Support

This script generates images of celebrities using HyperLoRA-modified Stable Diffusion,
with support for multi-GPU generation via torchrun.

Usage:
    # Single GPU
    python generate_images_celebrity_cfg.py --hypernetwork_path ./hyper_lora.pth --task 100 --output_dir ./celebrity_images

    # Multi-GPU (4 GPUs)
    torchrun --nproc_per_node=4 generate_images_celebrity_cfg.py --hypernetwork_path ./hyper_lora.pth --task 100 --output_dir ./celebrity_images
"""

import os
import re
import argparse
import time
import torch
import pandas as pd
import numpy as np
from pathlib import Path
from functools import partial
from typing import List, Tuple
from tqdm import tqdm

from transformers import CLIPTextModel, CLIPTokenizer
from torchvision.transforms.functional import to_pil_image

from hyper_lora import HyperLoRALinear, HypernetworkManager, inject_hyper_lora
from ldm.models.diffusion.ddimcopy import DDIMSampler
from utils import load_model_from_config, set_seed


PROMPT_TEMPLATES = [
    "A portrait of {name}",
    "An image capturing {name} at a public event",
    "An oil painting of {name}",
    "A sketch of {name}",
    "{name} in an official photo",
]

CSV_PATHS = {
    1: "prompts_csv/celebrity_1_concepts.csv",
    5: "prompts_csv/celebrity_5_concepts.csv",
    10: "prompts_csv/celebrity_10_concepts.csv",
    100: "prompts_csv/celebrity_100_concepts.csv",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate celebrity images with HyperLoRA and CFG",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--hypernetwork_path", type=str, required=True,
        help="Path to hypernetwork checkpoint (hyper_lora.pth)"
    )
    parser.add_argument(
        "--task", type=int, choices=[1, 5, 10, 100], default=100,
        help="Number of celebrities to generate (1, 5, 10, or 100)"
    )
    parser.add_argument(
        "--output_dir", type=str, required=True,
        help="Directory to save generated images"
    )
    parser.add_argument(
        "--config", type=str, default="./configs/stable-diffusion/v1-inference.yaml",
        help="Path to model config file"
    )
    parser.add_argument(
        "--ckpt", type=str, default="models/sd-v1-4.ckpt",
        help="Path to model checkpoint"
    )
    parser.add_argument(
        "--num_images", type=int, default=5,
        help="Number of images to generate per prompt"
    )
    parser.add_argument(
        "--steps", type=int, default=50,
        help="Number of DDIM sampling steps"
    )
    parser.add_argument(
        "--guidance_scale", type=float, default=7.5,
        help="CFG guidance scale"
    )
    parser.add_argument(
        "--image_size", type=int, default=512,
        help="Image size for generation"
    )
    parser.add_argument(
        "--eta", type=float, default=0.0,
        help="DDIM eta"
    )
    parser.add_argument(
        "--hyper_timestep", type=int, default=500,
        help="Timestep for HyperLoRA context"
    )
    parser.add_argument(
        "--alpha", type=float, default=0.00001,
        help="LoRA alpha scaling factor"
    )
    parser.add_argument(
        "--lora_rank", type=int, default=1,
        help="Rank of LoRA layers"
    )
    parser.add_argument(
        "--hidden_size", type=int, default=100,
        help="Hidden/Internal size for Hypernetwork"
    )
    parser.add_argument(
        "--seed", type=int, default=2024,
        help="Random seed base for reproducibility"
    )
    parser.add_argument(
        "--base_path", type=str, default=".",
        help="Base path for CSV files"
    )
    parser.add_argument(
        "--generate_type", type=str, choices=["erased", "others", "both"], default="both",
        help="Which celebrity types to generate: erased, others, or both"
    )
    return parser.parse_args()


class CombinedCFGModel:
    """Wrapper that uses the same model but toggles LoRA for unconditional/reference passes."""

    def __init__(self, model):
        self.model = model
        self.device = model.device

    def apply_model(self, x, t, c):
        b2 = x.shape[0]
        assert b2 % 2 == 0
        b = b2 // 2

        x_uncond = x[:b]
        x_cond = x[b:]
        t_uncond = t[:b]
        t_cond = t[b:]

        if isinstance(c, dict):
            c_uncond = {}
            c_cond = {}
            for k in c:
                if isinstance(c[k], list):
                    c_uncond[k] = [v[:b] for v in c[k]]
                    c_cond[k] = [v[b:] for v in c[k]]
                else:
                    c_uncond[k] = c[k][:b]
                    c_cond[k] = c[k][b:]
        else:
            c_uncond = c[:b]
            c_cond = c[b:]

        # Route unconditional to model without LoRA
        with self.model.hyper.no_lora():
            out_uncond = self.model.apply_model(x_uncond, t_uncond, c_uncond)

        # Route conditional to model with LoRA
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


def extract_name_from_prompt(prompt: str) -> str:
    for template in PROMPT_TEMPLATES:
        pattern = template.replace("{name}", "(.*)")
        match = re.match(pattern, prompt)
        if match:
            return match.group(1).strip()
    return prompt


def load_celebrity_lists(task: int, base_path: str = ".") -> Tuple[List[str], List[str]]:
    if task not in CSV_PATHS:
        raise ValueError(f"Invalid task: {task}. Must be one of {list(CSV_PATHS.keys())}")

    csv_path = Path(base_path) / CSV_PATHS[task]
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    df = pd.read_csv(csv_path)

    erased_df = df[df['type'] == 'erased']
    retained_df = df[df['type'] == 'others']

    erased_names = erased_df['prompt'].apply(extract_name_from_prompt).unique().tolist()
    retained_names = retained_df['prompt'].apply(extract_name_from_prompt).unique().tolist()

    return erased_names, retained_names


def generate_single_image(
    sampler,
    model,
    prompt: str,
    device: torch.device,
    steps: int = 50,
    eta: float = 0.0,
    start_code: torch.Tensor = None,
    guidance_scale: float = 7.5,
):
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


def main():
    # Multi-GPU setup
    LOCAL_RANK = int(os.environ.get("LOCAL_RANK", "0"))
    RANK = int(os.environ.get("RANK", 0))
    WORLD_SIZE = int(os.environ.get("WORLD_SIZE", 1))

    torch.cuda.set_device(LOCAL_RANK)
    device = torch.device(f"cuda:{LOCAL_RANK}")

    args = parse_args()

    print(f"[Rank {RANK}/{WORLD_SIZE}] Starting generation on {device}", flush=True)

    # Load celebrity lists
    erased_names, retained_names = load_celebrity_lists(args.task, args.base_path)

    if args.generate_type == "erased":
        all_names = [("erased", erased_names)]
    elif args.generate_type == "others":
        all_names = [("others", retained_names)]
    else:
        all_names = [("erased", erased_names), ("others", retained_names)]

    # Create output directories
    output_dir = Path(args.output_dir)
    for subdir, _ in all_names:
        (output_dir / subdir).mkdir(parents=True, exist_ok=True)

    # Load CLIP
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
    clip_text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14").to(device).eval()

    # Load model with HyperLoRA
    print(f"[Rank {RANK}] Loading model...", flush=True)
    model = load_model_from_config(args.config, args.ckpt, device)

    lora_sd = torch.load(args.hypernetwork_path, map_location=device)
    hyper_lora_factory = partial(
        HyperLoRALinear,
        clip_size=768,
        rank=args.lora_rank,
        train_steps=args.hyper_timestep,
        alpha=args.alpha,
        internal_size=args.hidden_size,
    )
    model.hyper = HypernetworkManager()
    hyper_lora_layers = inject_hyper_lora(
        model.model.diffusion_model, ["attn2.to_k", "attn2.to_v"], hyper_lora_factory
    )
    for layer_name, layer in hyper_lora_layers:
        layer.set_parent_model(model)
        model.hyper.add_hyperlora(layer_name, layer.hyper_lora)

    # Load weights
    sd = model.model.diffusion_model.state_dict()
    updated = 0
    with torch.no_grad():
        for k, v in lora_sd.items():
            if k in sd:
                if torch.is_tensor(lora_sd[k]) and torch.is_tensor(sd[k]) and lora_sd[k].shape == sd[k].shape:
                    sd[k].copy_(v.to(sd[k].dtype))
                    updated += 1

    print(f"[Rank {RANK}] Loaded {updated} tensors from checkpoint", flush=True)

    model.tokenizer = tokenizer
    model.clip_text_encoder = clip_text_encoder
    model.hyper_timestep = args.hyper_timestep

    # Create combined CFG model and sampler
    combined_model = CombinedCFGModel(model).eval()
    sampler = DDIMSampler(model=combined_model)

    # Build list of all generation tasks
    tasks = []
    for subdir, names in all_names:
        for name in names:
            for template in PROMPT_TEMPLATES:
                prompt = template.format(name=name)
                for seed_idx in range(1, args.num_images + 1):
                    name_for_file = name.replace(' ', '_')
                    prompt_for_file = template.format(name=name_for_file)
                    filename = f"{prompt_for_file}_{seed_idx}.png"
                    filepath = output_dir / subdir / filename
                    tasks.append((prompt, filepath, seed_idx, subdir))

    # Filter tasks for this rank
    my_tasks = [(p, fp, s, sd) for i, (p, fp, s, sd) in enumerate(tasks) if i % WORLD_SIZE == RANK]

    print(f"[Rank {RANK}] Processing {len(my_tasks)}/{len(tasks)} images", flush=True)

    # Generate images
    for prompt, filepath, seed_idx, subdir in tqdm(my_tasks, desc=f"[Rank {RANK}] Generating", disable=(RANK != 0)):
        if filepath.exists():
            continue

        seed = args.seed + seed_idx
        set_seed(seed)
        gen = torch.Generator(device=device).manual_seed(seed)

        start_code = torch.randn(
            1, 4, args.image_size // 8, args.image_size // 8,
            generator=gen, device=device
        )

        # Compute HyperLoRA context
        inputs = tokenizer(
            prompt,
            max_length=tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        ).to(device).input_ids

        t_prompt = clip_text_encoder(inputs).pooler_output.detach()
        timestep = torch.tensor([args.hyper_timestep]).to(device)
        model.hyper.set_context(t_prompt, timestep)
        model.hyper.compute_and_cache_loras(t_prompt, timestep)

        # Generate image
        img = generate_single_image(
            sampler=sampler,
            model=combined_model,
            prompt=prompt,
            device=device,
            steps=args.steps,
            eta=args.eta,
            start_code=start_code,
            guidance_scale=args.guidance_scale,
        )

        img_np = img[0].cpu().permute(1, 2, 0).numpy()
        img_pil = to_pil_image((img_np * 255).astype(np.uint8))
        img_pil.save(filepath)

    print(f"[Rank {RANK}] Done!", flush=True)


if __name__ == "__main__":
    main()
