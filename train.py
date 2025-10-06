#!/usr/bin/env python3
import argparse
import json
import os
from functools import partial
from pathlib import Path

import matplotlib.pyplot as plt  # (unused, but kept if you uncomment stats)
import numpy as np
import torch
import wandb
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed as hf_set_seed
from torch.utils.data import DataLoader
from torchvision.transforms.functional import to_pil_image
from transformers import CLIPTextModel, CLIPTokenizer
from tqdm import tqdm

from data_utils import TargetReferenceDataset, collate_prompts
from ldm.models.diffusion.ddimcopy import DDIMSampler
from ldm.util import instantiate_from_config  # (kept if your get_models uses it)
from hyper_lora import (HyperLoRALinear, inject_hyper_lora,
                        inject_hyper_lora_nsfw)
from sampling import sample_model
from utils import get_models, print_trainable_parameters  # DO NOT import set_seed here to avoid clashes


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="LoRA/HyperLoRA Fine-tuning for Stable Diffusion (Accelerate)"
    )

    # Model configuration
    parser.add_argument(
        "--config_path",
        type=str,
        default="./configs/stable-diffusion/v1-inference.yaml",
        help="Path to model configuration file",
    )
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default="./models/sd-v1-4.ckpt",
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--device", type=str, default="cuda:0",
        help="(Ignored when using Accelerate) Device to use for training"
    )

    # LoRA/HyperLoRA
    parser.add_argument("--lora_rank", type=int, default=1, help="LoRA rank parameter")
    parser.add_argument("--lora_alpha", type=float, default=8, help="LoRA alpha parameter")
    parser.add_argument(
        "--target_modules",
        nargs="+",
        default=["attn2.to_k", "attn2.to_v"],
        help="Target modules for LoRA injection",
    )
    parser.add_argument("--clip_size", type=int, default=768, help="CLIP embedding size")

    # Optim/Trainer
    parser.add_argument("--iterations", type=int, default=200, help="Number of training iterations")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--lr", type=float, default=3e-5, help="Learning rate")
    parser.add_argument("--image_size", type=int, default=512, help="Image size for training")
    parser.add_argument("--ddim_steps", type=int, default=50, help="DDIM sampling steps")
    parser.add_argument("--ddim_eta", type=float, default=0.0, help="DDIM eta")
    parser.add_argument("--start_guidance", type=float, default=9.0, help="Starting guidance scale")
    parser.add_argument("--negative_guidance", type=float, default=2.0, help="Negative guidance scale")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")

    # Logging / tracking
    parser.add_argument("--use-wandb", action="store_true", dest="use_wandb")
    parser.add_argument("--log_from", type=int, default=0, help="Log debug images from iteration")
    parser.add_argument(
        "--logging_dir", type=str, default="logs",
        help="Base logging directory (used by Accelerate trackers)."
    )
    parser.add_argument(
        "--mixed_precision", type=str, default=None, choices=["no", "fp16", "bf16"],
        help="Override Accelerate mixed precision (fp16/bf16)."
    )
    parser.add_argument(
        "--report_to", type=str, default="wandb",
        help='Tracking integration: "tensorboard", "wandb", "comet_ml", or "all".'
    )

    # Output / data
    parser.add_argument("--output_dir", type=str, default="output", help="Directory to save models")
    parser.add_argument("--data_dir", type=str, default="data100_curated", help="Directory with prompt json files")
    parser.add_argument("--save_losses", action="store_true", help="Save training losses to file")

    return parser.parse_args()


def create_quick_sampler(model, sampler, image_size: int, ddim_steps: int, ddim_eta: float):
    """Create a quick sampling function with fixed parameters"""
    return lambda conditioning, scale, start_code, till_T: sample_model(
        model,
        sampler,
        conditioning,
        image_size,
        image_size,
        ddim_steps,
        scale,
        ddim_eta,
        start_code=start_code,
        till_T=till_T,
        verbose=False,
    )


def generate_and_save_sd_images(
    model,
    sampler,
    prompt: str,
    device: torch.device,
    steps: int = 50,
    eta: float = 0.0,
    batch_size: int = 1,
    out_dir: str = "tmp",
    prefix: str = "unl_",
    start_code: torch.Tensor = None,   # optional noise tensor [B,4,64,64] for 512x512
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
    - out_dir: folder to save into
    - prefix: file prefix, e.g., 'unl_'
    - start_code: optional start noise shape [B, 4, H/8, W/8]; if None, sampled internally.
                  For 512×512 set shape to [B, 4, 64, 64].
    """
    if start_code is None:
        start_code = torch.randn(batch_size, 4, 64, 64, device=device)  # 512x512

    model.eval()
    with torch.no_grad(), torch.autocast(device_type=device.type, enabled=(device.type == "cuda")):
        cond   = model.get_learned_conditioning([prompt] * start_code.shape[0])
        uncond = model.get_learned_conditioning([""] * start_code.shape[0])

        samples_latent, _ = sampler.sample(
            S=steps,
            conditioning={"c_crossattn": [cond]},
            batch_size=start_code.shape[0],
            shape=start_code.shape[1:],  # (4, H/8, W/8)
            verbose=False,
            unconditional_guidance_scale=7.5,
            unconditional_conditioning={"c_crossattn": [uncond]},
            eta=eta,
            x_T=start_code,
        )

        imgs = model.decode_first_stage(samples_latent)       # [-1, 1]
        imgs = (imgs.clamp(-1, 1) + 1) / 2.0                 # [0, 1]

        out_path = Path(out_dir)
        out_path.mkdir(exist_ok=True, parents=True)
        for i, im in enumerate(imgs.cpu()):
            im_u8 = (im.clamp(0, 1) * 255).round().to(torch.uint8)  # [3,H,W]
            to_pil_image(im_u8).save(out_path / f"{prefix}{i:04d}.png")

        return imgs  # [B,3,H,W] in [0,1]


def main():
    args = parse_args()

    # Print basic config
    print("=== LoRA/HyperLoRA Fine-tuning (Accelerate) ===")
    print(f"Config: {args.config_path}")
    print(f"Checkpoint: {args.ckpt_path}")
    print(f"Output dir: {args.output_dir}")
    print(f"Iterations: {args.iterations}  |  LR: {args.lr}  |  Accum: {args.gradient_accumulation_steps}")
    print(f"Image size: {args.image_size}  |  DDIM steps: {args.ddim_steps}  |  eta: {args.ddim_eta}")
    print("=" * 48)

    # Seed
    if args.seed is not None:
        hf_set_seed(args.seed)

    # Accelerate project config
    accelerator_project_config = ProjectConfiguration(
        project_dir=args.output_dir,
        logging_dir=args.logging_dir,
    )

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,  # None -> use accelerate config
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )
    #logger = get_logger(__name__)
    is_main = accelerator.is_main_process

    # Trackers (W&B/TB/etc.) — initialize after Accelerator so it attaches run metadata
    if is_main and args.use_wandb and ("wandb" in str(args.report_to) or args.report_to == "all"):
        wandb.init(project="UnGuide", name="training", config=vars(args))

    # Data
    data_dir = args.data_dir
    ds = TargetReferenceDataset(data_dir)
    ds_loader = DataLoader(ds, batch_size=1, shuffle=True, collate_fn=collate_prompts)

    # Models (original + trainable clone)
    if is_main:
        os.makedirs(os.path.join(args.output_dir, "tmp"), exist_ok=True)

    model_orig, sampler_orig, model, sampler_unused = get_models(
        args.config_path, args.ckpt_path, accelerator.device
    )

    # Freeze original model
    for p in model_orig.model.diffusion_model.parameters():
        p.requires_grad = False
    model_orig.eval()

    # Add attribute used downstream
    model.current_conditioning = None

    # Inject HyperLoRA/LoRA BEFORE prepare(), then build optimizer on trainable params
    use_hyper = True  # your script forces hypernetwork on; keep same behavior
    hyper_lora_factory = partial(
        HyperLoRALinear,
        clip_size=args.clip_size,
        rank=args.lora_rank,
        alpha=args.lora_alpha,
    )
    hyper_lora_layers = inject_hyper_lora(
        model.model.diffusion_model, args.target_modules, hyper_lora_factory
    )
    for layer in hyper_lora_layers:
        layer.set_parent_model(model)

    # Optimizer on trainable (LoRA) params only
    trainable_params = list(filter(lambda p: p.requires_grad, model.model.diffusion_model.parameters()))
    if is_main:
        print(f"Total trainable parameter tensors: {len(trainable_params)}")
        print_trainable_parameters(model)

    optimizer = torch.optim.Adam(trainable_params, lr=args.lr)

    if is_main:
        print('Before prepare')

    # Prepare for DDP / Mixed precision
    model, optimizer, ds_loader = accelerator.prepare(model, optimizer, ds_loader)

    if is_main:
        print('After prepare')

    base = accelerator.unwrap_model(model)
    for layer in hyper_lora_layers:
        layer.set_parent_model(base)

        # Create sampler AFTER prepare so it uses the wrapped model
    sampler = DDIMSampler(model)

    # Tokenizer + CLIP text encoder (inference-only; keep unwrapped)
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
    clip_text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14").to(accelerator.device).eval()

    # Quick sampler
    quick_sampler = create_quick_sampler(model, sampler, args.image_size, args.ddim_steps, args.ddim_eta)

    # Optionally log a baseline image (main only)
    if is_main:
        imgs0 = generate_and_save_sd_images(
            model=model_orig,
            sampler=sampler,
            prompt=ds[0]["target"],
            device=accelerator.device,
            steps=50,
            out_dir=os.path.join(args.output_dir, "tmp"),
            prefix="orig_",
        )
        if args.use_wandb and imgs0 is not None:
            im0 = (imgs0[0].clamp(0, 1) * 255).round().to(torch.uint8).cpu()
            wandb.log({"baseline": wandb.Image(to_pil_image(im0))}, step=0)

    # Training
    criterion = torch.nn.MSELoss()
    losses = []

    pbar = tqdm(range(args.iterations), disable=not accelerator.is_local_main_process)
    if is_main:
        print('!!!!!!!!!!')
    for i in pbar:
        for sample_ids, sample in enumerate(ds_loader):
            # Get conditional embeddings (strings) directly for LDM
            emb_0 = model.get_learned_conditioning(sample["reference"])
            emb_p = model.get_learned_conditioning(sample["target"])
            emb_n = model.get_learned_conditioning(sample["target"])

            optimizer.zero_grad(set_to_none=True)

            # random timestep mapping (keep your logic)
            t_enc = torch.randint(args.ddim_steps, (1,), device=accelerator.device)
            og_num = round((int(t_enc) / args.ddim_steps) * 1000)
            og_num_lim = round((int(t_enc + 1) / args.ddim_steps) * 1000)
            t_enc_ddpm = torch.randint(og_num, og_num_lim, (1,), device=accelerator.device)

            # Build CLIP tokens for current target/reference (for HyperLoRA conditioning)
            def encode(text: str):
                return (
                    tokenizer(
                        text,
                        max_length=tokenizer.model_max_length,
                        padding="max_length",
                        truncation=True,
                        return_tensors="pt",
                    )
                    .to(accelerator.device)
                    .input_ids
                )

            inputs = (encode(sample["target"]), encode(sample["reference"]))
            with torch.no_grad():
                cond_target = clip_text_encoder(inputs[0]).pooler_output.detach()
                cond_ref    = clip_text_encoder(inputs[1]).pooler_output.detach()

            # pass both to model for HyperLoRA
            base = accelerator.unwrap_model(model)  # the actual Module used in forward
            base.current_conditioning = (cond_target, cond_ref)
            if is_main:
                print('!!! ', model.current_conditioning, base.current_conditioning)

            # starting latent code
            start_code = torch.randn(
                (1, 4, args.image_size // 8, args.image_size // 8),
                device=accelerator.device
            )

            with torch.no_grad():
                z  = quick_sampler(emb_p, args.start_guidance, start_code, int(t_enc))
                e_0 = model_orig.apply_model(z, t_enc_ddpm, emb_0)  # Reference
                e_p = model_orig.apply_model(z, t_enc_ddpm, emb_p)  # Target

            e_n = model.apply_model(z, t_enc_ddpm, emb_n)

            # targets and loss
            e_0.requires_grad = False
            e_p.requires_grad = False
            target = e_0 - (args.negative_guidance * (e_p - e_0))

            loss = criterion(e_n, target)

            # Backward with Accelerate
            accelerator.backward(loss)
            optimizer.step()

            # Optional image logging
            if (
                is_main
                and args.use_wandb
                and i >= args.log_from
                and i % 10 == 0
                and sample_ids == 0
            ):
                imgs = generate_and_save_sd_images(
                    model=model,
                    sampler=sampler,
                    prompt=sample["target"][0],
                    device=accelerator.device,
                    steps=50,
                    out_dir=os.path.join(args.output_dir, "tmp"),
                    prefix=f"unl_{i}_",
                )
                if imgs is not None:
                    caption = f"target: {sample['target'][0]}"
                    im0 = (imgs[0].clamp(0, 1) * 255).round().to(torch.uint8).cpu()
                    wandb.log({"sample": wandb.Image(to_pil_image(im0), caption=caption)}, step=i)

            loss_value = float(loss.detach().item())
            losses.append(loss_value)

            if is_main and args.use_wandb:
                wandb.log({"loss": loss_value, "iter": i}, step=i)

            if accelerator.is_local_main_process:
                pbar.set_postfix({"loss": f"{loss_value:.6f}"})

        # Save LoRA/HyperLoRA weights each iteration (or move outside loop if you prefer)
        if is_main:
            save_dir = os.path.join(args.output_dir, f"rank_{args.lora_rank}_it_{args.iterations}_lr_{args.lr}_sg_{args.start_guidance}_ng_{args.negative_guidance}_ddim_{args.ddim_steps}_" + ("hyper" if use_hyper else "lora"))
            os.makedirs(os.path.join(save_dir, "models"), exist_ok=True)

            lora_state_dict = {}
            # unwrap model for named_parameters if needed
            model_unwrapped = accelerator.unwrap_model(model)
            for name, param in model_unwrapped.model.diffusion_model.named_parameters():
                if param.requires_grad:
                    lora_state_dict[name] = param.detach().cpu().clone()

            model_filename = "hyper_lora.pth" if use_hyper else "lora.pth"
            lora_path = os.path.join(save_dir, "models", model_filename)
            accelerator.save(lora_state_dict, lora_path)

    # Wrap up
    if is_main:
        print("Training completed!")
        if losses:
            print(f"Final loss: {losses[-1]:.6f}")
            print(f"Average loss: {sum(losses)/len(losses):.6f}")

        # Dump config + basic metrics
        run_dir = os.path.dirname(lora_path) if losses else args.output_dir
        config = {
            "config": args.config_path,
            "ckpt": args.ckpt_path,
            "use_hypernetwork": use_hyper,
            "clip_size": args.clip_size,
            "lora_rank": args.lora_rank,
            "lora_alpha": args.lora_alpha,
            "target_modules": args.target_modules,
            "iterations": args.iterations,
            "lr": args.lr,
            "image_size": args.image_size,
            "ddim_steps": args.ddim_steps,
            "ddim_eta": args.ddim_eta,
            "start_guidance": args.start_guidance,
            "negative_guidance": args.negative_guidance,
            "final_loss": losses[-1] if losses else None,
            "average_loss": (sum(losses) / len(losses)) if losses else None,
        }
        with open(os.path.join(run_dir, "train_config.json"), "w") as f:
            json.dump(config, f, indent=2)

    if is_main and args.use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
