import argparse
import json
import os
import wandb
from functools import partial

import matplotlib.pyplot as plt
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from data_utils import TargetReferenceDataset, collate_prompts
from torchvision.transforms.functional import to_pil_image
import numpy as np
from transformers import CLIPTextModel, CLIPTokenizer
import torch
from tqdm import tqdm
from ldm.models.diffusion.ddimcopy import DDIMSampler

from hyper_lora import (HyperLoRALinear, inject_hyper_lora,
                        inject_hyper_lora_nsfw)
from ldm.util import instantiate_from_config
from sampling import sample_model
from utils import get_models, print_trainable_parameters, set_seed


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="LoRA/HyperLoRA Fine-tuning for Stable Diffusion"
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
        "--lora_alpha", type=float, default=8, help="LoRA alpha parameter"
    )
    parser.add_argument(
        "--target_modules",
        nargs="+",
        default=["attn2.to_k", "attn2.to_v"],
        help="Target modules for LoRA injection",
    )

    # HyperLoRA configuration
    # parser.add_argument(
    #     "--use_hypernetwork",
    #     action="store_true",
    #     help="Use HyperLoRA instead of regular LoRA",
    # )
    parser.add_argument(
        "--clip_size",
        type=int,
        default=768,
        help="CLIP embedding size",
    )
    parser.add_argument(
        "--data_file",
        type=str,
        default="embeddings.csv",
        help="A .csv file with prompt and embeddings",
    )
    # Training configuration
    parser.add_argument(
        "--iterations", type=int, default=200, help="Number of training iterations"
    )
    parser.add_argument("--use-wandb", action='store_true')
    parser.add_argument("--lr", type=float, default=3e-5, help="Learning rate")
    parser.add_argument(
        "--image_size", type=int, default=512, help="Image size for training"
    )
    parser.add_argument("--log_from", type=int, default=0, help="Log debug images from iteration")
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
    #parser.add_argument(
    #    "--prompts_json",
    #    type=str,
    #    default="data/cat.json",
    #    help="Path to JSON file containing prompts",
    #)
    parser.add_argument(
        "--save_losses", action="store_true", help="Save training losses to file"
    )

    return parser.parse_args()


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
        #cond   = model.get_learned_conditioning([prompt] * start_code.shape[0])
        uncond = model.get_learned_conditioning([""] * start_code.shape[0])

        samples_latent, _ = sampler.sample(
            S=steps,
            conditioning={"c_crossattn": [prompt]},
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


def main():
    args = parse_args()
    # temp
    args.use_hypernetwork = True

    print("=== LoRA/HyperLoRA Fine-tuning Configuration ===")
    print(f"Config: {args.config_path}")
    print(f"Checkpoint: {args.ckpt_path}")
    print(f"Device: {args.device}")
    print(f"Use HyperNetwork: {args.use_hypernetwork}")
    if args.use_hypernetwork:
        print(f"CLIP size: {args.clip_size}")
    print(f"LoRA rank: {args.lora_rank}, alpha: {args.lora_alpha}")
    print(f"Target modules: {args.target_modules}")
    print(f"Training iterations: {args.iterations}")
    print(f"Learning rate: {args.lr}")
    print("=" * 40)


    if args.use_wandb:
        wandb.init(
            project="UnGuide",
            name='training',
            group=None,
            config=vars(args)
        )

    # Set seed
    if args.seed is not None:
        set_seed(args.seed)


    data_dir = "data/"  # <-- change me
    ds = TargetReferenceDataset(data_dir)
    ds_loader = DataLoader(ds, batch_size=1, shuffle=True, collate_fn=collate_prompts)


    config = {
        "config": args.config_path,
        "ckpt": args.ckpt_path,
        "use_hypernetwork": args.use_hypernetwork,
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
    }

    lora_type = "hyper" if args.use_hypernetwork else "lora"
    dir_name = (
        "_".join(
            f"{k}_{config[k]}"
                for k in [
                "lora_rank",
                "iterations",
                "lr",
                "start_guidance",
                "negative_guidance",
                "ddim_steps",
            ]
        )
        + f"_{lora_type}"
    )
    os.makedirs(os.path.join(args.output_dir, dir_name, "models"), exist_ok=True)

    # Initialize models
    print("Loading models...")
    model_orig, sampler_orig, model, sampler = get_models(
        args.config_path, args.ckpt_path, args.device
    )

    # Freeze original model parameters
    for param in model.model.diffusion_model.parameters():
        param.requires_grad = False

    # Add current_conditioning attribute to model for HyperLoRA
    model.current_conditioning = None

    device = next(model.parameters()).device

    # --- sampling with CFG ---
    sampler = DDIMSampler(model)

    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
    clip_text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14").to(device).eval()

    inputs = tokenizer(
        ds[0]["target"],
        max_length=tokenizer.model_max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    ).to(args.device).input_ids

    t_prompt = clip_text_encoder(inputs).pooler_output.detach()

    model.current_conditioning = t_prompt
    print('!!! ', t_prompt.shape)
    generate_and_save_sd_images(
        model=model,
        sampler=sampler,
        prompt=t_prompt,
        device=device,
        steps=50,
        out_dir="tmp",
        prefix="orig_",
    )

    # Inject LoRA or HyperLoRA layers
    print(f"Injecting {'HyperLoRA' if args.use_hypernetwork else 'LoRA'} layers...")

    # Create HyperLoRA factory function
    hyper_lora_factory = partial(
        HyperLoRALinear,
        clip_size=args.clip_size,
        rank=args.lora_rank,
        alpha=args.lora_alpha,
    )

    #if args.prompts_json.endswith("nsfw.json"):
    #    hyper_lora_layers = inject_hyper_lora_nsfw(
    #        model.model.diffusion_model, hyper_lora_factory=hyper_lora_factory
    #    )
    #else:
    hyper_lora_layers = inject_hyper_lora(
        model.model.diffusion_model, args.target_modules, hyper_lora_factory
    )

    for layer in hyper_lora_layers:
        layer.set_parent_model(model)

    # Get trainable parameters (only HyperLoRA layers)
    trainable_params = list(
        filter(lambda p: p.requires_grad, model.model.diffusion_model.parameters())
    )
    print(f"Total trainable parameters: {len(trainable_params)}")
    print_trainable_parameters(model)

    # Set model to training mode
    model.train()

    # Disable checkpointing for faster training
    model.use_checkpoint = False
    model.model.diffusion_model.use_checkpoint = False

    # Initialize training components
    optimizer = torch.optim.Adam(trainable_params, lr=args.lr)
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
        for sample_ids, sample in enumerate(ds_loader):

            emb_0 = model.get_learned_conditioning(sample["reference"])
            emb_p = model.get_learned_conditioning(sample["target"])
            emb_n = model.get_learned_conditioning(sample["target"])

            optimizer.zero_grad()

            t_enc = torch.randint(args.ddim_steps, (1,), device=args.device)
            og_num = round((int(t_enc) / args.ddim_steps) * 1000)
            og_num_lim = round((int(t_enc + 1) / args.ddim_steps) * 1000)
            t_enc_ddpm = torch.randint(og_num, og_num_lim, (1,), device=args.device)

            inputs = tokenizer(
                sample["target"],
                max_length=tokenizer.model_max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            ).to(args.device).input_ids

            model.current_conditioning = clip_text_encoder(inputs).pooler_output.detach()
            model.current_conditioning.requires_grad = False

            start_code = torch.randn((1, 4, args.image_size // 8, args.image_size // 8)).to(
                args.device
            )

            with torch.no_grad():
                z = quick_sampler(emb_p, args.start_guidance, start_code, int(t_enc))
                e_0 = model_orig.apply_model(z, t_enc_ddpm, emb_0)  # Reference
                e_p = model_orig.apply_model(z, t_enc_ddpm, emb_p)  # Target
            e_n = model.apply_model(z, t_enc_ddpm, emb_n)

            e_0.requires_grad = False
            e_p.requires_grad = False

            target = e_0 - (args.negative_guidance * (e_p - e_0))
            loss = criterion(e_n, target)

            loss.backward()
            optimizer.step()

            if i >= args.log_from and i % 10 == 0 and args.use_wandb and sample_ids == 0:
                imgs = generate_and_save_sd_images(
                    model=model,
                    sampler=sampler,
                    prompt=model.current_conditioning,
                    device=device,
                    steps=50,
                    out_dir="tmp",
                    prefix=f"unl_{i}_",
                )

                caption = f"target: {sample['target'][0]}"
                im0 = (imgs[0].clamp(0, 1) * 255).round().to(torch.uint8).cpu()
                wandb.log(
                    {"sample": wandb.Image(im0, caption=caption)},
                    step=i,
                )

            loss_value = loss.item()
            if args.use_wandb:
                wandb.log({"loss": loss_value}, step=i)

            losses.append(loss_value)
            pbar.set_postfix({"loss": f"{loss_value:.6f}"})

        print(f"Saving trained model to {args.output_dir}/{dir_name}/models")
        model.current_conditioning = None
        lora_state_dict = {}
        for name, param in model.model.diffusion_model.named_parameters():
            if param.requires_grad:
                lora_state_dict[name] = param.cpu().detach().clone()

        model_filename = "hyper_lora.pth" if args.use_hypernetwork else "lora.pth"
        lora_path = os.path.join(args.output_dir, dir_name, "models", model_filename)
        torch.save(lora_state_dict, lora_path)

    # this part is unnecessary
    # print("Analyzing HyperLoRA weights...")
    # os.makedirs(os.path.join(args.output_dir, dir_name, "stats"), exist_ok=True)
    #
    # avg_weights = []
    # layer_names = []
    #
    # for i, layer in enumerate(hyper_lora_layers):
    #     hypernetwork = layer.hyper_lora.hypernetwork
    #     if hypernetwork is not None:
    #         weight_avg = hypernetwork.weight.data.abs().mean().item()
    #         bias_avg = hypernetwork.bias.data.abs().mean().item()
    #         avg_weights.append(weight_avg)
    #
    #         for name, module in model.model.diffusion_model.named_modules():
    #             if module is layer:
    #                 layer_names.append(name)
    #                 break
    #         else:
    #             layer_names.append(f"layer_{i}")
    #
    # if avg_weights and layer_names:
    #     plt.figure(figsize=(12, 8))
    #     plt.bar(range(len(avg_weights)), avg_weights, tick_label=layer_names)
    #     plt.xticks(rotation=90)
    #     plt.ylabel("Average Absolute Weight")
    #     plt.title("Average HyperLoRA Hypernetwork Weights")
    #     plt.tight_layout()
    #     plot_path = os.path.join(
    #         args.output_dir, dir_name, "stats", "hyperlora_weights.png"
    #     )
    #     plt.savefig(plot_path)
    #     print(f"Saved HyperLoRA weight plot to {plot_path}")
    #     plt.close()

    print("Training completed!")
    print(f"Final loss: {losses[-1]:.6f}")
    print(f"Average loss: {sum(losses) / len(losses):.6f}")

    if args.use_wandb:
        wandb.finish()

    config["final_loss"] = losses[-1]
    config["average_loss"] = sum(losses) / len(losses)

    with open(os.path.join(args.output_dir, dir_name, "train_config.json"), "w") as f:
        json.dump(config, f, indent=4)
    print(
        f"Training configuration saved to {os.path.join(args.output_dir, dir_name, 'train_config.json')}"
    )


if __name__ == "__main__":
    main()
