import os
import json
import argparse
from functools import partial


import numpy as np
import torch
import torch
import torch.nn as nn

from utils import print_trainable_parameters, set_seed, get_models
from tqdm import tqdm

import matplotlib.pyplot as plt
from pathlib import Path
from torchvision.transforms.functional import to_pil_image
import numpy as np
import torch
from tqdm import tqdm
from ldm.models.diffusion.ddimcopy import DDIMSampler


from ldm.util import instantiate_from_config
from sampling import sample_model
from lora import LoRALinear, inject_lora_nsfw, inject_lora

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="LoRA Fine-tuning for Stable Diffusion"
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
        "--device", type=str, default="cuda:0", help="Device to use for training"
    )

    # LoRA configuration
    parser.add_argument("--lora_rank", type=int, default=1, help="LoRA rank parameter")
    parser.add_argument(
        "--lora_alpha", type=int, default=8, help="LoRA alpha parameter"
    )
    parser.add_argument(
        "--target_modules",
        nargs="+",
        default=["attn2.to_k", "attn2.to_v"],
        help="Target modules for LoRA injection",
    )

    # Training configuration
    parser.add_argument(
        "--iterations", type=int, default=200, help="Number of training iterations"
    )
    parser.add_argument("--lr", type=float, default=3e-5, help="Learning rate")
    parser.add_argument(
        "--image_size", type=int, default=512, help="Image size for training"
    )
    parser.add_argument(
        "--ddim_steps", type=int, default=50, help="DDIM sampling steps"
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--ddim_eta", type=float, default=0.0, help="DDIM eta parameter"
    )
    parser.add_argument(
        "--start_guidance", type=float, default=9.0, help="Starting guidance scale"
    )
    parser.add_argument(
        "--negative_guidance", type=float, default=2.0, help="Negative guidance scale"
    )

    # Output
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output",
        help="Directory to save the trained model",
    )
    parser.add_argument(
        "--prompts_json",
        type=str,
        default="data/cat.json",
        help="Path to JSON file containing prompts",
    )
    parser.add_argument(
        "--save_losses", action="store_true", help="Save training losses to file"
    )

    return parser.parse_args()

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
    start_code: torch.Tensor = None,   # optional noise tensor [B,4,64,64] (for 512x512)
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
    - start_code: optional start noise of shape [B, 4, H/8, W/8]; if None, sampled internally.
                  For 512×512 set shape to [B, 4, 64, 64].
    """
    # derive latent shape from start_code or default to 512×512
    if start_code is None:
        start_code = torch.randn(batch_size, 4, 64, 64, device=device)  # 512x512

    # freeze & eval for safety

    with torch.no_grad(), torch.autocast(device_type=device.type, enabled=(device.type == "cuda")):
        cond   = model.get_learned_conditioning([prompt] * start_code.shape[0])
        uncond = model.get_learned_conditioning([""] * start_code.shape[0])

        samples_latent, _ = sampler.sample(
            S=steps,
            conditioning={"c_crossattn": [cond]},
            batch_size=start_code.shape[0],
            shape=start_code.shape[1:],  # (4, H/8, W/8)
            verbose=False,
            unconditional_guidance_scale=7.5,                 # CFG scale; tweak if needed
            unconditional_conditioning={"c_crossattn": [uncond]},
            eta=eta,
            x_T=start_code,
        )

        # decode latents to [0,1] images
        imgs = model.decode_first_stage(samples_latent)       # [-1, 1]
        imgs = (imgs.clamp(-1, 1) + 1) / 2.0                 # [0, 1]

        # save
        out_path = Path(out_dir)
        out_path.mkdir(exist_ok=True)
        for i, im in enumerate(imgs.cpu()):
            im_u8 = (im.clamp(0, 1) * 255).round().to(torch.uint8)  # [3,H,W]
            to_pil_image(im_u8).save(out_path / f"{prefix}{i:04d}.png")

        print(f"Saved {len(imgs)} image(s) to {out_path}/ with prefix '{prefix}'")
        return imgs  # [B,3,H,W] in [0,1]


def create_quick_sampler(
    model, sampler, image_size: int, ddim_steps: int, ddim_eta: float
):
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


def main():
    args = parse_args()

    print("=== LoRA Fine-tuning Configuration ===")
    print(f"Config: {args.config_path}")
    print(f"Checkpoint: {args.ckpt_path}")
    print(f"Device: {args.device}")
    print(f"LoRA rank: {args.lora_rank}, alpha: {args.lora_alpha}")
    print(f"Target modules: {args.target_modules}")
    print(f"Training iterations: {args.iterations}")
    print(f"Learning rate: {args.lr}")
    print("=" * 40)
    
    # Set seed
    if args.seed is not None:
        set_seed(args.seed)

    # Load prompts json
    with open(args.prompts_json, 'r') as f:
        prompts_data = json.load(f)

    target_prompt = prompts_data.get("target", None)
    reference_prompt = prompts_data.get("reference", None)
    if target_prompt is None or reference_prompt is None:
        raise ValueError(f"Missing required prompt")
    
    config = {  
        "config": args.config_path,
        "ckpt": args.ckpt_path,
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
        "target_prompt": target_prompt,
        "reference_prompt": reference_prompt,
        "prompts_json": args.prompts_json,
        "class_name": args.prompts_json.split("/")[-1].split(".")[0],
    }
    
    dir_name =  "_".join(f"{k}_{config[k]}" for k in [
        "class_name",
        "lora_rank",
        "lora_alpha",
        "iterations",
        "lr",
        "start_guidance",
        "negative_guidance",
        "ddim_steps"
    ])
    os.makedirs(os.path.join(args.output_dir, dir_name, "models"), exist_ok=True)

    # Initialize models
    print("Loading models...")
    model_orig, sampler_orig, model, sampler = get_models(
        args.config_path, args.ckpt_path, args.device
    )

    # Create LoRA factory function
    lora_factory = partial(LoRALinear, rank=args.lora_rank, alpha=args.lora_alpha)
    # Freeze original model parameters
    for param in model.model.diffusion_model.parameters():
        param.requires_grad = False

    # --- sampling with CFG ---
    sampler = DDIMSampler(model)

    device = next(model.parameters()).device

    generate_and_save_sd_images(
        model=model,
        sampler=sampler,
        prompt=target_prompt,
        device=device,
        steps=50,
        out_dir="tmp_main",
        prefix="orig_",
    )


    # Inject LoRA layers
    print("Injecting LoRA layers...")
    if args.prompts_json.endswith("nsfw.json"):
        inject_lora_nsfw(model.model.diffusion_model, lora_factory=lora_factory)
    else:
        inject_lora(model.model.diffusion_model, args.target_modules, lora_factory)

    # Get trainable parameters (only LoRA layers)
    lora_layers = list(
        filter(lambda p: p.requires_grad, model.model.diffusion_model.parameters())
    )
    print(f"Total trainable parameters: {len(lora_layers)}")
    print_trainable_parameters(model)

    # Set model to training mode
    model.train()

    # Disable checkpointing for faster training
    model.use_checkpoint = False
    model.model.diffusion_model.use_checkpoint = False

    # Initialize training components
    optimizer = torch.optim.Adam(lora_layers, lr=args.lr)
    criterion = torch.nn.MSELoss()
    losses = []

    # Create quick sampling function
    quick_sampler = create_quick_sampler(
        model, sampler, args.image_size, args.ddim_steps, args.ddim_eta
    )

    # Training loop
    print("Starting training...")
    pbar = tqdm(range(args.iterations))

    for i in pbar:
        # Prepare embeddings
        emb_0 = model.get_learned_conditioning([reference_prompt])
        emb_p = model.get_learned_conditioning([target_prompt])
        emb_n = model.get_learned_conditioning([target_prompt])

        optimizer.zero_grad()

        # Sample random timestep
        t_enc = torch.randint(args.ddim_steps, (1,), device=args.device)
        og_num = round((int(t_enc) / args.ddim_steps) * 1000)
        og_num_lim = round((int(t_enc + 1) / args.ddim_steps) * 1000)
        t_enc_ddpm = torch.randint(og_num, og_num_lim, (1,), device=args.device)

        # Generate random starting noise
        start_code = torch.randn((1, 4, args.image_size // 8, args.image_size // 8)).to(
            args.device
        )

        with torch.no_grad():
            # Sample latent representation
            z = quick_sampler(emb_p, args.start_guidance, start_code, int(t_enc))
            
            # Get predictions from original model
            e_0 = model_orig.apply_model(z, t_enc_ddpm, emb_0)  # Reference
            e_p = model_orig.apply_model(z, t_enc_ddpm, emb_p)  # Target

        # Get prediction from trainable model
        e_n = model.apply_model(z, t_enc_ddpm, emb_n)

        # Ensure gradients are not computed for reference predictions
        e_0.requires_grad = False
        e_p.requires_grad = False

        # Compute loss (negative guidance objective)
        target = e_0 - (args.negative_guidance * (e_p - e_0))
        loss = criterion(e_n, target)

        # Backward pass and optimization
        loss.backward()

        def gather_trainable_params_and_names_from_model(model: nn.Module):
            params, name_map = [], {}
            for name, p in model.named_parameters(recurse=True):
                if p.requires_grad:
                    params.append(p)
                    name_map[id(p)] = name
                    print(f"[TRAINABLE] {name}  shape={tuple(p.shape)}")  # <- prints names
            print(f"Total trainable tensors: {len(params)}")
            return params, name_map

        # Build once (after you construct lora_layers)
        params, name_map = gather_trainable_params_and_names_from_model(model)

        # ----- manual SGD step with prints -----
        lr = args.lr

        updated_any = False
        with torch.no_grad():
            for p in params:
                g = p.grad
                if g is None:
                    continue
                # print BEFORE modifying/clearing grads
                pname = name_map.get(id(p), "<unknown>")
                try:
                    gnorm = g.norm().item()
                except Exception:
                    gnorm = float('nan')
                print(f"UPDATE {pname}: ||grad||={gnorm:.3e}", flush=True)

                # θ ← θ − lr * ∇θ
                p.add_(g, alpha=-lr)
                updated_any = True

        if not updated_any:
            print("No params updated (all grads None).", flush=True)

        # clear gradients
        for p in params:
            p.grad = None

        #optimizer.step()

        generate_and_save_sd_images(
            model=model,
            sampler=sampler,
            prompt=target_prompt,
            device=device,
            steps=50,
            out_dir="tmp_main",
            prefix=f"unl_{i}_",
        )

        generate_and_save_sd_images(
            model=model,
            sampler=sampler,
            prompt="car",
            device=device,
            steps=50,
            out_dir="tmp_main",
            prefix=f"car_unl_{i}_",
        )

        # Record loss
        loss_value = loss.item()
        losses.append(loss_value)
        pbar.set_postfix({"loss": f"{loss_value:.6f}"})

    # Save trained model
    
    print(f"Saving trained model to {args.output_dir}/{dir_name}/models")
    lora_state_dict = {}
    for name, param in model.model.diffusion_model.named_parameters():
        if param.requires_grad:
            lora_state_dict[name] = param.cpu().detach().clone()
    lora_path = os.path.join(args.output_dir, dir_name, "models", "lora.pth")
    torch.save(lora_state_dict, lora_path)

    
    print("Training completed!")
    print(f"Final loss: {losses[-1]:.6f}")
    print(f"Average loss: {sum(losses) / len(losses):.6f}")
    
    config["final_loss"] = losses[-1]
    config["average_loss"] = sum(losses) / len(losses)
    
    with open(os.path.join(args.output_dir, dir_name, "train_config.json"), 'w') as f:
        json.dump(config, f, indent=4)
    print(f"Training configuration saved to {os.path.join(args.output_dir, 'train_config.json')}")
    


if __name__ == "__main__":
    main()

