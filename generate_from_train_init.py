#!/usr/bin/env python3
"""
Generation script that initializes model EXACTLY like train_simple.py,
then loads checkpoint weights and generates images.

Default (no --csv): Celebrity mode - generates from celebrity CSV, splits into erased/others
With --csv: COCO mode - generates from provided CSV, no splitting

Usage:
    # Celebrity generation (default)
    accelerate launch generate_from_train_init.py --config configs/celebrity/train_celebrity_100_nvembed.yaml --checkpoint hyper_lora_5999.pth

    # COCO generation (with --csv)
    accelerate launch generate_from_train_init.py --config configs/celebrity/train_celebrity_100_nvembed.yaml --checkpoint hyper_lora_5999.pth --csv mscoco_30k.csv

    # Multi-GPU
    accelerate launch --num_processes=4 generate_from_train_init.py --config configs/celebrity/train_celebrity_100_nvembed.yaml --checkpoint hyper_lora_5999.pth
"""

import argparse
import os
from functools import partial
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
import yaml
from accelerate import Accelerator
from accelerate.utils import set_seed as hf_set_seed
from torchvision.transforms.functional import to_pil_image
from tqdm import tqdm

from hyper_lora import HyperLoRALinear, HypernetworkManager, inject_hyper_lora
from ldm.models.diffusion.ddimcopy import DDIMSampler
from utils import load_model_from_config
from nv_embed_utils import load_nv_embed_model, NV_EMBED_DIM, NV_EMBED_MODEL_NAME, NV_EMBED_INSTRUCTION
from nemo_neva_utils import load_neva_model, compute_neva_embed, NEVA_EMBED_DIM, NEVA_MODEL_NAME


# Hardcoded paths
CELEBRITY_CSV_PATH = "prompts_csv/celebrity_100_concepts.csv"
CELEBRITY_OUTPUT_DIR = "./generated_celebrity"
COCO_OUTPUT_DIR = "./generated_coco"


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

        with self.model.hyper.no_lora():
            out_uncond = self.model.apply_model(x_uncond, t_uncond, c_uncond)

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


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    config_name = list(config.keys())[0]
    return config[config_name], config_name


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate images with model initialized exactly like train_simple.py"
    )
    parser.add_argument(
        "--config", type=str, required=True,
        help="Path to YAML configuration file (same as used for training)"
    )
    parser.add_argument(
        "--checkpoint", type=str, default="hyper_lora_5999.pth",
        help="Path to checkpoint file"
    )
    parser.add_argument(
        "--csv", type=str, default=None,
        help="Path to COCO CSV file. If not provided, uses celebrity mode."
    )
    parser.add_argument(
        "--steps", type=int, default=50,
        help="Number of DDIM steps"
    )
    parser.add_argument(
        "--guidance_scale", type=float, default=7.5,
        help="CFG guidance scale"
    )
    parser.add_argument(
        "--hyper_timestep", type=int, default=None,
        help="HyperLoRA timestep (defaults to hyper_train_steps from config)"
    )
    parser.add_argument(
        "--generate_type", type=str, choices=["erased", "others", "both"], default="both",
        help="(Celebrity mode only) Which types to generate: erased, others, or both"
    )
    parser.add_argument(
        "--max_images", type=int, default=None,
        help="Maximum number of images to generate (for testing)"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Determine mode
    is_coco_mode = args.csv is not None

    # Load configuration - EXACTLY like train_simple.py
    config, config_name = load_config(args.config)

    # Extract parameters - EXACTLY like train_simple.py
    rank_lora = config.get('rank', 1)
    lora_alpha = config.get('lora_alpha', 8)
    internal_size = config.get('internal_size', 100)
    seed = config.get('seed', 2024)
    resolution = config.get('resolution', 512)
    use_orig_concat = config.get('use_orig_concat', False)
    hyper_train_steps = config.get('hyper_train_steps', 500)
    gradient_accumulation_steps = config.get('gradient_accumulation_steps', 1)

    embedding_model = config.get('embedding_model', 'nv_embed')

    pretrained_model_path = config.get('pretrained_model_name_or_path', './models/sd-v1-4.ckpt')
    model_config_path = config.get('model_config', './configs/stable-diffusion/v1-inference.yaml')

    hyper_timestep = args.hyper_timestep if args.hyper_timestep is not None else hyper_train_steps

    # Set seed BEFORE accelerator for determinism - EXACTLY like train_simple.py
    if seed is not None:
        hf_set_seed(seed)

    # Initialize accelerator
    accelerator = Accelerator(
        gradient_accumulation_steps=gradient_accumulation_steps,
        mixed_precision=config.get('mixed_precision', None),
    )

    device = accelerator.device
    is_main = accelerator.is_main_process
    rank = accelerator.process_index
    world_size = accelerator.num_processes

    if is_main:
        print(f"=== Generating with config: {config_name} ===")
        print(f"Mode: {'COCO' if is_coco_mode else 'Celebrity'}")
        print(f"Config file: {args.config}")
        print(f"Checkpoint: {args.checkpoint}")
        print(f"World size: {world_size}")
        print(f"Seed: {seed}")
        print(f"LoRA rank: {rank_lora}")
        print(f"LoRA alpha: {lora_alpha}")
        print(f"Internal size: {internal_size}")
        print(f"Hyper train steps: {hyper_train_steps}")
        print(f"Hyper timestep for generation: {hyper_timestep}")
        print(f"use_orig_concat: {use_orig_concat}")
        print(f"Embedding model: {embedding_model}")
        print("=" * 48)

    # Load model - EXACTLY like train_simple.py
    if is_main:
        print("Loading base model...")
    model = load_model_from_config(model_config_path, pretrained_model_path, device)

    # Freeze backbone - EXACTLY like train_simple.py
    for p in model.model.diffusion_model.parameters():
        p.requires_grad = False

    # Setup HyperLoRA - EXACTLY like train_simple.py
    if is_main:
        print("Setting up HyperLoRA...")
    model.hyper = HypernetworkManager()

    if embedding_model == 'neva':
        clip_size = NEVA_EMBED_DIM
        embed_model_name = NEVA_MODEL_NAME
    else:
        clip_size = NV_EMBED_DIM
        embed_model_name = NV_EMBED_MODEL_NAME

    if is_main:
        print(f"Using {embed_model_name} with embedding dim {clip_size}")

    target_modules = ["attn2.to_k", "attn2.to_v"]

    # HyperLoRA factory - EXACTLY like train_simple.py
    hyper_lora_factory = partial(
        HyperLoRALinear,
        clip_size=clip_size,
        rank=rank_lora,
        alpha=lora_alpha,
        train_steps=hyper_train_steps,
        use_orig_concat=use_orig_concat,
        internal_size=internal_size,
    )

    hyper_lora_layers = inject_hyper_lora(
        model.model.diffusion_model, target_modules, hyper_lora_factory
    )

    for layer_name, layer in hyper_lora_layers:
        layer.set_parent_model(model)
        model.hyper.add_hyperlora(layer_name, layer.hyper_lora)

    # ============================================================
    # LOAD THE CHECKPOINT
    # ============================================================
    if is_main:
        print(f"\nLoading checkpoint from {args.checkpoint}...")
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    if is_main:
        print(f"Checkpoint has {len(checkpoint)} parameter tensors")

    # Load state dict into model
    missing, unexpected = model.model.diffusion_model.load_state_dict(checkpoint, strict=False)
    if is_main:
        print(f"Missing keys: {len(missing)}")
        print(f"Unexpected keys: {len(unexpected)}")

        if len(unexpected) > 0:
            print(f"WARNING: Unexpected keys in checkpoint: {unexpected[:5]}...")

        # Verify some alpha values loaded correctly
        print("\nVerifying loaded alpha values:")
        alpha_count = 0
        for name, param in model.model.diffusion_model.named_parameters():
            if 'hyper_lora.alpha' in name:
                print(f"  {name}: {param.item():.6f}")
                alpha_count += 1
                if alpha_count >= 3:
                    break

    # ============================================================
    # LOAD EMBEDDING MODEL
    # ============================================================
    if is_main:
        print(f"\nLoading {embed_model_name}...")
    if embedding_model == 'neva':
        embed_model, embed_tokenizer = load_neva_model(device, torch.float16)
    else:
        embed_model, embed_tokenizer = load_nv_embed_model(device, torch.float16)

    # Flag for which embedding interface to use
    use_neva_embed = (embedding_model == 'neva')

    # ============================================================
    # LOAD CSV AND SETUP OUTPUT
    # ============================================================
    if is_coco_mode:
        # COCO mode
        csv_path = args.csv
        if is_main:
            print(f"\nLoading COCO CSV from {csv_path}...")
        df = pd.read_csv(csv_path)

        # Output dir: ./generated_coco/{csv_name_without_extension}/
        csv_name = Path(csv_path).stem
        output_dir = Path(COCO_OUTPUT_DIR) / csv_name
        # All processes create directory (exist_ok=True makes this safe)
        output_dir.mkdir(parents=True, exist_ok=True)
        if is_main:
            print(f"Output directory: {output_dir}")

        # Scan existing files and extract IDs (handles both old and new patterns)
        # Old pattern: {case_num}_{prompt}_{seed}.png
        # New pattern: {case_num}.png
        existing_ids = set()
        for f in output_dir.glob("*.png"):
            # Extract ID from filename (first part before _ or before .png)
            fname = f.stem  # filename without extension
            case_id = fname.split('_')[0]
            try:
                existing_ids.add(int(case_id))
            except ValueError:
                pass  # skip files that don't start with a number

        if is_main and existing_ids:
            print(f"Found {len(existing_ids)} existing images, will skip those IDs")

        # Build task list: (prompt, filepath, eval_seed)
        tasks = []
        skipped = 0
        for idx, row in df.iterrows():
            prompt = row['prompt']
            eval_seed = int(row['evaluation_seed'])
            case_num = row['case_number'] if 'case_number' in df.columns else idx

            # Skip if this ID already has an image
            if int(case_num) in existing_ids:
                skipped += 1
                continue

            filename = f"{case_num}.png"
            filepath = output_dir / filename
            tasks.append((prompt, filepath, eval_seed))

        if is_main and skipped:
            print(f"Skipped {skipped} already generated images")

    else:
        # Celebrity mode
        csv_path = CELEBRITY_CSV_PATH
        if is_main:
            print(f"\nLoading Celebrity CSV from {csv_path}...")
        df = pd.read_csv(csv_path)

        # Filter by type
        if args.generate_type == "erased":
            df = df[df['type'] == 'erased']
        elif args.generate_type == "others":
            df = df[df['type'] == 'others']

        output_dir = Path(CELEBRITY_OUTPUT_DIR)
        # All processes create directories (exist_ok=True makes this safe)
        (output_dir / "erased").mkdir(parents=True, exist_ok=True)
        (output_dir / "others").mkdir(parents=True, exist_ok=True)
        if is_main:
            print(f"Output directory: {output_dir}")

        # Build task list: (prompt, filepath, eval_seed)
        tasks = []
        for idx, row in df.iterrows():
            prompt = row['prompt']
            img_type = row['type']
            eval_seed = int(row['evaluation_seed'])

            safe_prompt = prompt.replace(' ', '_').replace(',', '').replace("'", "").replace('"', '')[:80]
            filename = f"{safe_prompt}_{eval_seed}.png"
            filepath = output_dir / img_type / filename
            tasks.append((prompt, filepath, eval_seed))

    if is_main:
        print(f"Total tasks: {len(tasks)}")

    if args.max_images is not None:
        tasks = tasks[:args.max_images]
        if is_main:
            print(f"Limited to {len(tasks)} images")

    # Wait for all processes
    accelerator.wait_for_everyone()

    # Distribute tasks across GPUs
    my_tasks = [t for i, t in enumerate(tasks) if i % world_size == rank]
    print(f"[Rank {rank}] Processing {len(my_tasks)}/{len(tasks)} images")

    # Create sampler with CFG wrapper
    combined_model = CombinedCFGModel(model).eval()
    sampler = DDIMSampler(model=combined_model)

    # ============================================================
    # GENERATION LOOP
    # ============================================================
    pbar = tqdm(my_tasks, desc=f"[Rank {rank}] Generating", disable=(rank != 0))

    for prompt, filepath, eval_seed in pbar:
        # Skip if already exists
        if filepath.exists():
            continue

        # Set seed for this image - deterministic
        image_seed = seed + eval_seed
        hf_set_seed(image_seed)
        torch.manual_seed(image_seed)
        torch.cuda.manual_seed_all(image_seed)

        # Create generator for deterministic noise
        generator = torch.Generator(device=device).manual_seed(image_seed)

        # Compute embedding for prompt
        if use_neva_embed:
            # NEVA uses compute_neva_embed function
            embedding = compute_neva_embed(
                [prompt], embed_model, embed_tokenizer, device, batch_size=1
            )
            embedding = embedding.to(dtype=torch.float32)
            # compute_neva_embed already returns normalized embeddings
        else:
            # NV-Embed uses model.encode() method
            embedding = embed_model.encode(
                [prompt],
                instruction=NV_EMBED_INSTRUCTION,
                max_length=4096,
            )
            if not isinstance(embedding, torch.Tensor):
                embedding = torch.tensor(embedding, device=device, dtype=torch.float32)
            else:
                embedding = embedding.to(device=device, dtype=torch.float32)
            embedding = F.normalize(embedding, p=2, dim=-1)

        # Set HyperLoRA context
        timestep = torch.tensor([hyper_timestep], device=device)
        model.hyper.set_context(embedding, timestep)
        model.hyper.compute_and_cache_loras(embedding, timestep)

        # Generate deterministic start code
        start_code = torch.randn(
            1, 4, resolution // 8, resolution // 8,
            generator=generator, device=device
        )

        with torch.no_grad(), torch.autocast(device_type=device.type, enabled=(device.type == "cuda")):
            cond = model.get_learned_conditioning([prompt])
            uncond = model.get_learned_conditioning([""])

            samples, _ = sampler.sample(
                S=args.steps,
                conditioning={"c_crossattn": [cond]},
                batch_size=1,
                shape=start_code.shape[1:],
                verbose=False,
                unconditional_guidance_scale=args.guidance_scale,
                unconditional_conditioning={"c_crossattn": [uncond]},
                eta=0.0,
                x_T=start_code,
            )
            decoded = model.decode_first_stage(samples)
            decoded = (decoded + 1.0) / 2.0
            decoded = torch.clamp(decoded, 0.0, 1.0)

        # Save image
        img = to_pil_image(decoded[0].cpu())
        img.save(filepath)

    accelerator.wait_for_everyone()

    if is_main:
        print(f"\nDone! Images saved to {output_dir}")


if __name__ == "__main__":
    main()
