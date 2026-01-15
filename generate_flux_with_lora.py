#!/usr/bin/env python3
"""
Generate images with FLUX model + HyperLoRA weights.
Loads trained HyperLoRA weights and applies them to the base FLUX model.

Usage:
    python generate_flux_with_lora.py \
        --lora_path output_nudity_flux/LoRA_fusion_model/hyper_lora_999.pth \
        --csv_path data/I2P_prompts_4703.csv \
        --output_dir generated_flux_lora
"""

import os
import argparse
import torch
import pandas as pd
from tqdm import tqdm
import time
import re
from functools import partial
from diffusers import FluxPipeline, FluxTransformer2DModel
from transformers import CLIPTextModel, CLIPTokenizer
from huggingface_hub import login

from hyper_lora import HyperLoRALinear, HypernetworkManager, inject_hyper_lora
from flux_model_wrapper import FluxModelWrapper

# HuggingFace token setup
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
token = os.environ.get("HF_TOKEN")
if token:
    login(token)
else:
    print("Warning: HF_TOKEN not set.")


def coerce_prompt(v):
    """Convert various prompt formats to clean string."""
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


def load_lora_weights(model_wrapper, lora_path, device):
    """
    Load HyperLoRA weights from saved checkpoint and apply to model.

    Args:
        model_wrapper: FluxModelWrapper with HyperLoRA injected
        lora_path: Path to saved LoRA weights (.pth file)
        device: Device to load weights to
    """
    print(f"Loading LoRA weights from: {lora_path}")

    if not os.path.exists(lora_path):
        raise FileNotFoundError(f"LoRA checkpoint not found: {lora_path}")

    lora_state_dict = torch.load(lora_path, map_location=device)
    print(f"Found {len(lora_state_dict)} trainable parameters in checkpoint")

    transformer = model_wrapper.transformer
    sd = transformer.state_dict()

    updated = 0
    skipped = []

    with torch.no_grad():
        for k, v in lora_state_dict.items():
            if k in sd:
                if torch.is_tensor(lora_state_dict[k]) and torch.is_tensor(v) and lora_state_dict[k].shape == v.shape:
                    sd[k].copy_(v.to(sd[k].dtype).to(device))
                    updated += 1
                else:
                    skipped.append((k, "shape/dtype mismatch"))
            else:
                skipped.append((k, "no such key in model"))

    print(f"[LoRA] Copied {updated} tensors, skipped {len(skipped)}")
    if skipped and len(skipped) <= 10:
        print(f"Skipped keys: {[k for k, reason in skipped]}")

    return model_wrapper


def setup_hyperlora_context(model_wrapper, target_emb, hyper_timestep, device, dtype):
    """
    Set HyperLoRA context for generation.

    Args:
        model_wrapper: FluxModelWrapper with HyperLoRA
        target_emb: CLIP embedding for the concept
        hyper_timestep: Which hypernetwork timestep to use (0 to hyper_train_steps)
        device: Device
        dtype: Data type
    """
    if model_wrapper.hyper is not None:
        model_wrapper.hyper.set_context(
            target_emb.to(device=device, dtype=dtype),
            torch.tensor([hyper_timestep], device=device, dtype=dtype)
        )
        model_wrapper.hyper.compute_and_cache_loras(
            target_emb.to(device=device, dtype=dtype),
            torch.tensor([hyper_timestep], device=device, dtype=dtype)
        )


def main():
    parser = argparse.ArgumentParser(description="Generate images with FLUX + HyperLoRA")
    parser.add_argument("--lora_path", type=str, required=True,
                       help="Path to saved LoRA weights (.pth file)")
    parser.add_argument("--csv_path", type=str, default="data/I2P_prompts_4703.csv",
                       help="CSV file with prompts")
    parser.add_argument("--output_dir", type=str, default="generated_flux_lora",
                       help="Output directory for generated images")
    parser.add_argument("--save_folder", type=str, default="images",
                       help="Subfolder name for images")
    parser.add_argument("--image_size", type=int, default=512,
                       help="Image size (height and width)")
    parser.add_argument("--num_inference_steps", type=int, default=28,
                       help="Number of inference steps")
    parser.add_argument("--guidance_scale", type=float, default=3.0,
                       help="Guidance scale for generation")
    parser.add_argument("--n_images", type=int, default=None,
                       help="Max number of images to generate (None = all)")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device to use")

    # HyperLoRA configuration
    parser.add_argument("--rank", type=int, default=9,
                       help="LoRA rank (must match training config)")
    parser.add_argument("--lora_alpha", type=float, default=9.0,
                       help="LoRA alpha (must match training config)")
    parser.add_argument("--hyper_train_steps", type=int, default=300,
                       help="Hypernetwork timesteps (must match training config)")
    parser.add_argument("--hyper_timestep", type=int, default=0,
                       help="Which hypernetwork timestep to use for generation (0 to hyper_train_steps)")
    parser.add_argument("--use_pooler", type=bool, default=True,
                       help="Use CLIP pooler output")
    parser.add_argument("--use_orig_concat", type=bool, default=False,
                       help="Use original concat in HyperLoRA")

    # Optional: concept-based context
    parser.add_argument("--context_concept", type=str, default=None,
                       help="Optional: CLIP text for HyperLoRA context (e.g., 'nudity')")

    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    weight_dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    print("=" * 60)
    print("Loading FLUX model with HyperLoRA")
    print("=" * 60)

    # Load base FLUX components
    pretrained_model_path = "black-forest-labs/FLUX.1-dev"
    cache_dir = "./models"
    os.makedirs(cache_dir, exist_ok=True)

    print(f"Loading FLUX transformer from {pretrained_model_path}...")
    transformer = FluxTransformer2DModel.from_pretrained(
        pretrained_model_path,
        torch_dtype=weight_dtype,
        subfolder="transformer",
        cache_dir=cache_dir,
        revision=None,
        variant=None
    ).to(device)

    transformer.requires_grad_(False)

    print("Loading text encoders and VAE...")
    pipe_temp = FluxPipeline.from_pretrained(
        pretrained_model_path,
        torch_dtype=weight_dtype,
        cache_dir=cache_dir
    )

    vae = pipe_temp.vae.to(device)
    text_encoder_one = pipe_temp.text_encoder.to(device)
    text_encoder_two = pipe_temp.text_encoder_2.to(device)
    tokenizer_one = pipe_temp.tokenizer
    tokenizer_two = pipe_temp.tokenizer_2

    text_encoders = [text_encoder_one, text_encoder_two]
    tokenizers = [tokenizer_one, tokenizer_two]

    print("Creating FluxModelWrapper...")
    model_wrapper = FluxModelWrapper(
        transformer=transformer,
        text_encoders=text_encoders,
        tokenizers=tokenizers,
        vae=vae,
        vae_scale_factor=2 ** (len(vae.config.block_out_channels)),
        max_sequence_length=512,
        device=device,
        dtype=weight_dtype
    )

    print("Setting up HyperLoRA...")
    model_wrapper.hyper = HypernetworkManager()

    clip_size = 768 if args.use_pooler else 512
    target_modules = ["attn.add_k_proj", "attn.add_q_proj"]

    hyper_lora_factory = partial(
        HyperLoRALinear,
        clip_size=clip_size,
        rank=args.rank,
        alpha=args.lora_alpha,
        train_steps=args.hyper_train_steps,
        use_orig_concat=args.use_orig_concat,
        dtype=weight_dtype,
    )

    hyper_lora_layers = inject_hyper_lora(
        transformer, target_modules, hyper_lora_factory
    )

    for layer_name, layer in hyper_lora_layers:
        layer.set_parent_model(model_wrapper)
        layer.to(dtype=weight_dtype)
        model_wrapper.hyper.add_hyperlora(layer_name, layer.hyper_lora)

    print(f"Injected HyperLoRA into {len(hyper_lora_layers)} layers")

    load_lora_weights(model_wrapper, args.lora_path, device)

    context_emb = None
    if args.context_concept:
        print(f"Setting up HyperLoRA context with concept: '{args.context_concept}'")
        clip_tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
        clip_text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14").to(device).eval()

        inputs = clip_tokenizer(
            args.context_concept,
            max_length=clip_tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        ).to(device).input_ids

        with torch.no_grad():
            if args.use_pooler:
                context_emb = clip_text_encoder(inputs).pooler_output.detach()
            else:
                context_emb = clip_text_encoder(inputs).last_hidden_state.detach()

        setup_hyperlora_context(
            model_wrapper,
            context_emb,
            args.hyper_timestep,
            device,
            weight_dtype
        )

    print("Creating generation pipeline...")
    pipe = FluxPipeline(
        transformer=transformer,
        vae=vae,
        scheduler=pipe_temp.scheduler,
        text_encoder=text_encoder_one,
        tokenizer=tokenizer_one,
        text_encoder_2=text_encoder_two,
        tokenizer_2=tokenizer_two,
    )
    pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=True)

    print(f"Loading prompts from {args.csv_path}...")
    df = pd.read_csv(args.csv_path, index_col=0)

    save_dir = os.path.join(args.output_dir, args.save_folder)
    os.makedirs(save_dir, exist_ok=True)

    print("=" * 60)
    print(f"Starting generation: {len(df)} prompts")
    print(f"Output directory: {save_dir}")
    print(f"HyperLoRA timestep: {args.hyper_timestep}/{args.hyper_train_steps}")
    print("=" * 60)

    images_generated = 0
    for image_id, row in tqdm(df.iterrows(), total=len(df)):
        if args.n_images is not None and images_generated >= args.n_images:
            break

        image_path = os.path.join(save_dir, f"{image_id:05d}.png")
        if os.path.exists(image_path):
            continue

        prompt = coerce_prompt(row.get("prompt", ""))
        if not isinstance(prompt, str) or not prompt.strip():
            print(f"Skip [{image_id}] empty prompt")
            continue

        seed = int(row.get("evaluation_seed", 0))
        generator = torch.Generator(device).manual_seed(seed)

        start = time.time()

        with torch.no_grad():
            image = pipe(
                prompt=prompt,
                guidance_scale=args.guidance_scale,
                num_inference_steps=args.num_inference_steps,
                height=args.image_size,
                width=args.image_size,
                generator=generator,
                max_sequence_length=256
            ).images[0]

        image.save(image_path)
        images_generated += 1
        end = time.time()

        if images_generated % 10 == 0:
            print(f"Generated {images_generated} images...")

    print("=" * 60)
    print(f"Generation complete! {images_generated} images saved to {save_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
