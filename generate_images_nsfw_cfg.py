import os
import json
import argparse
import torch
import re
import pandas as pd
import numpy as np
from tqdm import tqdm
import time
from functools import partial
from torchvision.utils import save_image
from torchvision.transforms.functional import to_pil_image

from ldm.models.diffusion.ddimcopy import DDIMSampler
from transformers import CLIPTextModel, CLIPTokenizer
from utils import load_model_from_config, set_seed
from hyper_lora import HyperLoRALinear, HypernetworkManager, inject_hyper_lora


class CombinedCFGModel:
    """Wrapper that uses different models for conditional and unconditional passes."""

    def __init__(self, cond_model, uncond_model):
        self.cond_model = cond_model
        self.uncond_model = uncond_model
        self.device = cond_model.device

    def apply_model(self, x, t, c):
        # When DDIMSampler uses guidance, it concatenates [uncond, cond] inputs
        # We split and route to different models
        b2 = x.shape[0]
        assert b2 % 2 == 0
        b = b2 // 2

        x_uncond = x[:b]
        x_cond = x[b:]
        t_uncond = t[:b]
        t_cond = t[b:]

        # Split conditioning
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

        # Route unconditional to model_orig, conditional to model (with LoRA)
        out_uncond = self.uncond_model.apply_model(x_uncond, t_uncond, c_uncond)
        out_cond = self.cond_model.apply_model(x_cond, t_cond, c_cond)

        return torch.cat([out_uncond, out_cond], dim=0)

    def get_learned_conditioning(self, prompts):
        return self.cond_model.get_learned_conditioning(prompts)

    def decode_first_stage(self, z):
        return self.cond_model.decode_first_stage(z)

    def eval(self):
        self.cond_model.eval()
        self.uncond_model.eval()
        return self

    def __getattr__(self, name):
        return getattr(self.cond_model, name)


def generate_images(
    sampler,
    model,
    prompt: str,
    device: torch.device,
    steps: int = 50,
    eta: float = 0.0,
    batch_size: int = 1,
    start_code: torch.Tensor = None,
    guidance_scale: float = 7.5,
):
    """
    Generates images with CFG from a CompVis SD model + DDIMSampler and saves them.

    - model: Stable Diffusion model (CompVis LDM style)
    - sampler: DDIMSampler(model)
    - prompt: text prompt
    - device: torch.device("cuda") or torch.device("cpu")
    - steps: DDIM steps
    - eta: DDIM eta (0.0 => deterministic)
    - batch_size: number of samples to generate
    - start_code: optional start noise shape [B, 4, H/8, W/8]; if None, sampled internally.
                  For 512×512 set shape to [B, 4, 64, 64].
    - guidance_scale: unconditional guidance scale
    """
    if start_code is None:
        start_code = torch.randn(batch_size, 4, 64, 64, device=device)

    model.eval()
    with torch.no_grad(), torch.autocast(device_type=device.type, enabled=(device.type == "cuda")):
        cond = model.get_learned_conditioning([prompt] * start_code.shape[0])
        uncond = model.get_learned_conditioning([""] * start_code.shape[0])

        samples, _ = sampler.sample(
            S=steps,
            conditioning={"c_crossattn": [cond]},
            batch_size=start_code.shape[0],
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


def coerce_prompt(v):
    # Treat None/NaN as empty
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return ""

    # Real list/tuple/set -> comma-separated string
    if isinstance(v, (list, tuple, set)):
        return ", ".join(str(x).strip() for x in v if str(x).strip())

    # String cases
    s = str(v).strip()

    # Drop an optional "Prompt" label like "Prompt [a, b]" or "Prompt: a, b"
    s = re.sub(r'^\s*prompt\s*[:\-]?\s*', "", s, flags=re.I)

    # If it's bracketed like "[a, b]" without quotes, normalize it
    m = re.match(r'^\[\s*(.*)\s*\]$', s)
    if m:
        parts = [p.strip() for p in m.group(1).split(",")]
        return ", ".join(p for p in parts if p)

    return s


if __name__ == "__main__":
    RANK = int(os.environ.get("RANK", 0))
    WORLD_SIZE = int(os.environ.get("WORLD_SIZE", 1))

    parser = argparse.ArgumentParser(
        description="Generate images with dynamic LoRA guidance weight"
    )
    parser.add_argument("--csv_path", type=str, default="I2P_prompts_4703.csv")
    parser.add_argument("--output_dir", type=str, default="generated_i2p")
    parser.add_argument(
        "--config", type=str, default="configs/stable-diffusion/v1-inference.yaml"
    )
    parser.add_argument("--ckpt", type=str, default="models/sd-v1-4.ckpt")
    parser.add_argument(
        "--alpha", type=float, default=0.00001, help="LoRA alpha scaling factor"
    )
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--ddim_steps", type=int, default=50)
    parser.add_argument("--nudity", type=bool, default=true)
    parser.add_argument("--ddim_eta", type=float, default=0.0)
    parser.add_argument("--guidance_scale", type=float, default=7.5, help="Default guidance scale (fallback if not in CSV)")
    parser.add_argument("--hyper_timestep", type=int, default=500, help="Timestep for HyperLoRA context")
    parser.add_argument("--n_images", type=int, default=None, help="Number of images to generate (if None, generate all)")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()
    print("Start", flush=True)
    print(f"Using device: {args.device}")
    if os.path.exists(os.path.join(args.output_dir, "train_config.json")):
        dirs = [args.output_dir]
    else:
        dirs = os.listdir(args.output_dir)

    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
    clip_text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14").to(args.device).eval()

    for dirname in dirs:
        # Load prompts
        df = pd.read_csv(args.csv_path, index_col=0)
        
        # Check if this is an NSFW dataset with nudity_percentage column
        if args.nudity and "nudity_percentage" in df.columns:
            # ensure numeric (coerce bad values to NaN)
            df["nudity_percentage"] = pd.to_numeric(df["nudity_percentage"], errors="coerce")
            # keep rows with nudity_percentage > 0
            df = df[df["nudity_percentage"].gt(0)]
            # sort descending
            df = df.sort_values(by="nudity_percentage", ascending=False)
        
        exp_dirpath = args.output_dir
        os.makedirs(os.path.join(exp_dirpath, args.output_dir), exist_ok=True)
        lora_path = os.path.join(exp_dirpath, "LoRA_fusion_model", "hyper_lora.pth")
        print(lora_path)

        if not os.path.exists(lora_path):
            print(f"Skip {dirname} - hyper_lora.pth not found")
            continue
        if len(os.listdir(os.path.join(exp_dirpath, "images"))) >= len(df):
            print(f"Skip {dirname} - already processed")
            continue
        print(f"Processing experiment: {dirname}.", flush=True)
        print("images", len(os.listdir(os.path.join(exp_dirpath, "images"))), flush=True)

        # Load and prepare models - model_orig for unconditional, model for conditional
        model_orig = load_model_from_config(args.config, args.ckpt, args.device)
        model = load_model_from_config(args.config, args.ckpt, args.device)

        # Apply HyperLoRA to conditional model only
        lora_sd = torch.load(lora_path, map_location=args.device)
        hyper_lora_factory = partial(
            HyperLoRALinear,
            clip_size=768,
            rank=1,
            train_steps=args.hyper_timestep,
            alpha=0.00001,
        )
        model.hyper = HypernetworkManager()
        hyper_lora_layers = inject_hyper_lora(
            model.model.diffusion_model, ["attn2.to_k", "attn2.to_v"], hyper_lora_factory
        )
        for layer_name, layer in hyper_lora_layers:
            layer.set_parent_model(model)
            model.hyper.add_hyperlora(layer_name, layer.hyper_lora)

        updated = 0
        skipped = []

        sd = model.model.diffusion_model.state_dict()

        with torch.no_grad():
            for k, v in lora_sd.items():
                if k in sd:
                    if torch.is_tensor(lora_sd[k]) and torch.is_tensor(v) and lora_sd[k].shape == v.shape:
                        sd[k].copy_(v.to(sd[k].dtype))
                        updated += 1
                        print("updated:", k)
                    else:
                        skipped.append((k, "shape/dtype mismatch"))
                else:
                    skipped.append((k, "no such key in model"))

        print(f"[LoRA] copied {updated} tensors, skipped {len(skipped)}")

        # Iterate over prompts
        images_generated = 0
        for image_id, row in df.iterrows():
            if args.n_images is not None and images_generated >= args.n_images:
                break
            image_path = os.path.join(exp_dirpath, "images", f"{image_id:05d}.jpg")
            if os.path.exists(image_path):
                continue  # Skip if image already exists
            
            if image_id % WORLD_SIZE != RANK:
                continue
            
            prompt = coerce_prompt(row.get("prompt", ""))
            if not isinstance(prompt, str) or not prompt.strip():
                print(f"Skip [{image_id}] empty prompt")
                continue
            start = time.time()
            seed = int(row.get("evaluation_seed", image_id))
            guidance = float(row.get("evaluation_guidance", args.guidance_scale))
            set_seed(seed)

            print(prompt)

            start_code = torch.randn(
                (1, 4, args.image_size // 8, args.image_size // 8),
                device=model.device
            )

            inputs = tokenizer(
                prompt,
                max_length=tokenizer.model_max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            ).to(model.device).input_ids

            t_prompt = clip_text_encoder(inputs).pooler_output.detach()

            model.hyper.set_context(t_prompt, torch.tensor([args.hyper_timestep]).to(model.device))
            model.hyper.compute_and_cache_loras(t_prompt, torch.tensor([args.hyper_timestep]).to(model.device))

            # Use combined model: conditional uses model (with LoRA), unconditional uses model_orig
            combined_model = CombinedCFGModel(cond_model=model, uncond_model=model_orig).eval()
            sampler = DDIMSampler(model=combined_model)
            
            img = generate_images(
                sampler=sampler, model=combined_model,
                start_code=start_code, prompt=prompt, device=model.device,
                steps=args.ddim_steps, guidance_scale=guidance
            )

            img_np = img[0].cpu().permute(1, 2, 0).numpy()
            img_pil = to_pil_image((img_np * 255).astype(np.uint8))

            img_pil.save(image_path)
            images_generated += 1
            end = time.time()
            print(f"Prompt [{prompt}] processed in {end - start:.2f}) seconds. Saved to {image_path}", flush=True)
