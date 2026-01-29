#!/usr/bin/env python3
"""
Generate celebrity images from CSV prompts using trained LoRA model.
Based on MACE evaluation protocol.
"""

import argparse
import torch
import os
import pandas as pd
from functools import partial
from hyper_lora import HyperLoRALinear, HypernetworkManager, inject_hyper_lora
from ldm.models.diffusion.ddimcopy import DDIMSampler
from utils import load_model_from_config, set_seed
from torchvision.transforms.functional import to_pil_image
import numpy as np
from tqdm import tqdm

# Import embedding models
from nv_embed_utils import load_nv_embed_model, compute_nv_embed
from nemo_neva_utils import load_neva_model, compute_neva_embed


def load_lora_weights(model, lora_path, device):
    """
    Load HyperLoRA weights from saved checkpoint and apply to model.

    Args:
        model: Stable Diffusion model with HyperLoRA injected
        lora_path: Path to saved LoRA weights (.pth file)
        device: Device to load weights to
    """
    print(f"Loading LoRA weights from: {lora_path}")

    if not os.path.exists(lora_path):
        raise FileNotFoundError(f"LoRA checkpoint not found: {lora_path}")

    lora_state_dict = torch.load(lora_path, map_location=device)
    print(f"Found {len(lora_state_dict)} trainable parameters in checkpoint")

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
                    skipped.append((k, f"shape mismatch: model={sd[k].shape}, ckpt={v.shape}"))
            else:
                skipped.append((k, "no such key in model"))

    print(f"[LoRA] Copied {updated} tensors, skipped {len(skipped)}")
    if skipped and len(skipped) <= 10:
        print(f"Skipped keys: {[k for k, reason in skipped]}")

    return model


def generate_images(sampler, model, prompt, device, steps=50, eta=0.0,
                   start_code=None, guidance_scale=7.5):
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


def main():
    parser = argparse.ArgumentParser(
        description='Generate celebrity images from CSV using HyperLoRA (MACE protocol)'
    )
    parser.add_argument('--lora-path', type=str, required=True,
                       help='Path to LoRA_fusion_model directory (contains hyper_lora.pth)')
    parser.add_argument('--prompts-csv', type=str, required=True,
                       help='CSV file with columns: type, prompt, evaluation_seed')
    parser.add_argument('--output-dir', type=str, required=True,
                       help='Directory to save generated images')
    parser.add_argument('--config', type=str, default='./configs/stable-diffusion/v1-inference.yaml',
                       help='Path to model config file')
    parser.add_argument('--ckpt', type=str, default='models/sd-v1-4.ckpt',
                       help='Path to model checkpoint')
    parser.add_argument('--steps', type=int, default=50,
                       help='Number of DDIM sampling steps')
    parser.add_argument('--guidance-scale', type=float, default=7.5,
                       help='CFG guidance scale')
    parser.add_argument('--ddim-eta', type=float, default=0.0,
                       help='DDIM eta parameter')
    parser.add_argument('--image-size', type=int, default=512,
                       help='Image size for generation')
    # ⚠️ CRITICAL: These must match training config!
    parser.add_argument('--hyper-timestep', type=int, default=300,
                       help='Timestep for HyperLoRA context (must match hyper_train_steps from training)')
    parser.add_argument('--alpha', type=float, default=1.0,
                       help='LoRA alpha scaling factor (must match lora_alpha from training config)')
    parser.add_argument('--lora-rank', type=int, default=6,
                       help='LoRA rank (must match training config)')
    parser.add_argument('--hidden-size', type=int, default=512,
                       help='HyperLoRA internal size (must match training config)')
    parser.add_argument('--embedding-model', type=str, default='nvembed', choices=['nvembed', 'neva'],
                       help='Embedding model type (must match training)')
    parser.add_argument('--device', type=str, default='cuda:0',
                       help='Device to run generation on')
    parser.add_argument('--init-buffers', type=str, default=None,
                       help='Path to initialization_buffers.pth file (xL_const_flat, xR_const_flat)')

    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # CRITICAL: Set seed before model creation so xL_const_flat/xR_const_flat
    # are initialized consistently (they're not saved in old checkpoints)
    set_seed(2024)  # Use the same seed as training!

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Reading prompts from {args.prompts_csv}...")
    df = pd.read_csv(args.prompts_csv)

    if 'prompt' not in df.columns or 'evaluation_seed' not in df.columns:
        raise ValueError("CSV must have 'prompt' and 'evaluation_seed' columns")

    print(f"Found {len(df)} prompts to generate")

    # Load embedding model (NV-Embed or NeVA, NOT CLIP!)
    print(f"\nLoading embedding model: {args.embedding_model}...")
    if args.embedding_model == 'neva':
        embed_model, embed_tokenizer = load_neva_model(device=device, dtype=torch.float16)
        compute_embed_fn = compute_neva_embed
    else:
        embed_model, embed_tokenizer = load_nv_embed_model(device=device, dtype=torch.float16)
        compute_embed_fn = compute_nv_embed
    
    clip_size = 4096  # Both NV-Embed and NeVA use 4096-dim embeddings

    print("Loading Stable Diffusion model...")
    model = load_model_from_config(args.config, args.ckpt, device)

    # Find the checkpoint file
    lora_file = args.lora_path
    if os.path.isdir(args.lora_path):
        # Try to find the latest checkpoint
        candidates = [f for f in os.listdir(args.lora_path) if f.startswith('hyper_lora') and f.endswith('.pth')]
        if candidates:
            candidates.sort()
            lora_file = os.path.join(args.lora_path, candidates[-1])
        else:
            lora_file = os.path.join(args.lora_path, "hyper_lora.pth")
    
    # Create HyperLoRA with CORRECT parameters matching training
    hyper_lora_factory = partial(
        HyperLoRALinear,
        clip_size=clip_size,  # 4096 for NV-Embed/NeVA, NOT 768!
        rank=args.lora_rank,  # 6, NOT 1!
        train_steps=args.hyper_timestep,  # 300
        alpha=args.alpha,  # 1.0
        internal_size=args.hidden_size,  # 512
    )

    model.hyper = HypernetworkManager()
    hyper_lora_layers = inject_hyper_lora(
        model.model.diffusion_model, ["attn2.to_k", "attn2.to_v"], hyper_lora_factory
    )
    for layer_name, layer in hyper_lora_layers:
        layer.set_parent_model(model)
        model.hyper.add_hyperlora(layer_name, layer.hyper_lora)

    print(f"✓ Injected {len(hyper_lora_layers)} HyperLoRA layers")

    # Load initialization buffers if provided (xL_const_flat, xR_const_flat)
    if args.init_buffers is not None:
        if os.path.exists(args.init_buffers):
            print(f"\n[LOADING INITIALIZATION BUFFERS]")
            print(f"Loading from: {args.init_buffers}")
            init_buffer_dict = torch.load(args.init_buffers, map_location=device)
            print(f"Found {len(init_buffer_dict)} buffers in file")

            loaded_count = 0
            for name, module in model.model.diffusion_model.named_modules():
                if hasattr(module, 'hyper_lora'):
                    xL_key = f"{name}.hyper_lora.xL_const_flat"
                    xR_key = f"{name}.hyper_lora.xR_const_flat"

                    if xL_key in init_buffer_dict:
                        module.hyper_lora.xL_const_flat.copy_(init_buffer_dict[xL_key].to(device))
                        loaded_count += 1
                    if xR_key in init_buffer_dict:
                        module.hyper_lora.xR_const_flat.copy_(init_buffer_dict[xR_key].to(device))
                        loaded_count += 1

            print(f"✓ Loaded {loaded_count} initialization buffers")
        else:
            print(f"WARNING: Initialization buffers file not found: {args.init_buffers}")

    # Print alpha values BEFORE loading checkpoint
    print("\n[BEFORE LOADING] Alpha values:")
    alpha_count = 0
    for name, param in model.model.diffusion_model.named_parameters():
        if 'hyper_lora.alpha' in name:
            if alpha_count < 3:
                print(f"  {name} = {param.item():.6f}")
            alpha_count += 1
    print(f"  Total: {alpha_count} alpha parameters")

    # Load checkpoint using manual tensor copy (same as Flux version)
    load_lora_weights(model, lora_file, device)

    # Print alpha values AFTER loading checkpoint
    print("\n[AFTER LOADING] Alpha values:")
    alpha_count = 0
    for name, param in model.model.diffusion_model.named_parameters():
        if 'hyper_lora.alpha' in name:
            if alpha_count < 3:
                print(f"  {name} = {param.item():.6f}")
            alpha_count += 1
    print(f"  Total: {alpha_count} alpha parameters")

    # Check if alphas in checkpoint
    print("\n[CHECKPOINT] Checking for alpha keys:")
    lora_sd = torch.load(lora_file, map_location=device)
    alpha_in_ckpt = 0
    for k in lora_sd.keys():
        if 'hyper_lora.alpha' in k:
            if alpha_in_ckpt < 3:
                print(f"  {k} = {lora_sd[k].item():.6f}")
            alpha_in_ckpt += 1
    print(f"  Total: {alpha_in_ckpt} alpha parameters in checkpoint")

    # Use single model with LoRA toggle for CFG (like training diagnostics)
    class CombinedCFGModelSingleModel:
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

    combined_model = CombinedCFGModelSingleModel(model).eval()
    sampler = DDIMSampler(model=combined_model)

    print("\nGenerating images...")
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Generating"):
        prompt = row['prompt']
        seed = int(row['evaluation_seed'])

        set_seed(seed)
        gen = torch.Generator(device=device).manual_seed(seed)

        start_code = torch.randn(
            1, 4, args.image_size // 8, args.image_size // 8,
            generator=gen, device=device
        )

        # Compute embedding using NV-Embed or NeVA (NOT CLIP!)
        t_prompt = compute_embed_fn(
            [prompt],
            embed_model,
            embed_tokenizer,
            device,
            batch_size=1
        )

        # DIAGNOSTIC: Print embedding info for first iteration
        if idx == 0:
            print(f"\n[EMBEDDING DIAGNOSTIC]")
            print(f"  Prompt: {prompt}")
            print(f"  Embedding shape: {t_prompt.shape}")
            print(f"  Embedding dtype: {t_prompt.dtype}")
            print(f"  Embedding device: {t_prompt.device}")
            print(f"  Embedding norm: {torch.norm(t_prompt).item():.6f}")
            print(f"  Embedding mean: {t_prompt.mean().item():.6f}")
            print(f"  Embedding std: {t_prompt.std().item():.6f}")
            print(f"  Embedding min/max: {t_prompt.min().item():.6f} / {t_prompt.max().item():.6f}")

        # Get the correct device and dtype for HyperLoRA layers (like Flux version does)
        hyper_device = model.hyper.hyper_layers[0].alpha.device if model.hyper.hyper_layers else device
        weight_dtype = model.hyper.hyper_layers[0].alpha.dtype if model.hyper.hyper_layers else torch.float32

        # Ensure embeddings and timestep are on correct device and dtype
        t_prompt = t_prompt.to(dtype=weight_dtype, device=hyper_device)
        timestep = torch.tensor([args.hyper_timestep], dtype=weight_dtype, device=hyper_device)

        if idx == 0:
            print(f"\n[DTYPE/DEVICE CONVERSION]")
            print(f"  HyperLoRA device: {hyper_device}")
            print(f"  HyperLoRA dtype: {weight_dtype}")
            print(f"  Embedding device after conversion: {t_prompt.device}")
            print(f"  Embedding dtype after conversion: {t_prompt.dtype}")
            print(f"  Timestep device: {timestep.device}")
            print(f"  Timestep dtype: {timestep.dtype}")

        # Set HyperLoRA context and compute LoRA weights
        model.hyper.set_context(t_prompt, timestep)
        model.hyper.compute_and_cache_loras(t_prompt, timestep)

        # DIAGNOSTIC: Verify context was set correctly
        if idx == 0:
            ctx_emb, ctx_time = model.hyper.get_context()
            print(f"\n[CONTEXT DIAGNOSTIC]")
            print(f"  Context embedding shape: {ctx_emb.shape if ctx_emb is not None else None}")
            print(f"  Context timestep: {ctx_time.item() if ctx_time is not None else None}")
            print(f"  LoRA enabled: {model.hyper.lora_enabled}")
            print(f"  Cached LoRA layers: {len(model.hyper.lora_weights_cache)}")

            # Check if cached LoRA weights are non-zero
            if len(model.hyper.lora_weights_cache) > 0:
                first_key = list(model.hyper.lora_weights_cache.keys())[0]
                alpha, x_L, x_R = model.hyper.lora_weights_cache[first_key]
                print(f"  Sample LoRA alpha: {alpha.item():.6f}")
                print(f"  Sample LoRA x_L norm: {torch.norm(x_L).item():.6f}")
                print(f"  Sample LoRA x_R norm: {torch.norm(x_R).item():.6f}")

        with torch.no_grad():
            # DIAGNOSTIC: Check context before generation
            if idx == 0:
                ctx_emb_before, ctx_time_before = model.hyper.get_context()
                print(f"\n[BEFORE GENERATION]")
                print(f"  Context still set: {ctx_emb_before is not None}")
                print(f"  Cached LoRAs: {len(model.hyper.lora_weights_cache)}")

            img = generate_images(
                sampler=sampler,
                model=combined_model,
                start_code=start_code,
                prompt=prompt,
                device=model.device,
                steps=args.steps,
                guidance_scale=args.guidance_scale,
                eta=args.ddim_eta
            )

            # DIAGNOSTIC: Check context after generation
            if idx == 0:
                ctx_emb_after, ctx_time_after = model.hyper.get_context()
                print(f"\n[AFTER GENERATION]")
                print(f"  Context still set: {ctx_emb_after is not None}")
                print(f"  Cached LoRAs: {len(model.hyper.lora_weights_cache)}")

        img_np = img[0].cpu().permute(1, 2, 0).numpy()
        img_pil = to_pil_image((img_np * 255).astype(np.uint8))

        filename = f"{prompt}_{seed}.png"
        filepath = os.path.join(args.output_dir, filename)
        img_pil.save(filepath, format='PNG')

    print(f"\nAll images generated successfully! Saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
