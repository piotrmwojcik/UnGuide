import os
import json
import argparse
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
import time
from functools import partial
from torchvision.utils import save_image
from torchvision.transforms.functional import to_pil_image

from generate_images import decide_w, load_model_from_config, AutoGuidedModel
from ldm.models.diffusion.ddimcopy import DDIMSampler
from transformers import CLIPTextModel, CLIPTokenizer
from sampling import sample_model
from utils import apply_lora_to_model, set_seed
from hyper_lora import HyperLoRALinear, HypernetworkManager, inject_hyper_lora


def generate_images(
    sampler,
    model,
    prompt: str,
    device: torch.device,
    steps: int = 50,
    eta: float = 0.0,
    batch_size: int = 1,
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

        samples, _ = sampler.sample(
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
        decoded = model.decode_first_stage(samples)
        decoded = (decoded + 1.0) / 2.0
        decoded = torch.clamp(decoded, 0.0, 1.0)
        return decoded  # [B,3,H,W] in [0,1]



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
        "--alpha", type=float, default=8.0, help="LoRA alpha scaling factor"
    )
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--ddim_steps", type=int, default=50)
    parser.add_argument(
        "--t_enc", type=int, default=40, help="Timestep at which to compute latent diff"
    )
    parser.add_argument("--ddim_eta", type=float, default=0.0)
    parser.add_argument("--batch_size", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=30)
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
        exp_dirpath = os.path.join(args.output_dir, dirname)
        os.makedirs(os.path.join(exp_dirpath, "images"), exist_ok=True)
        lora_path = os.path.join(exp_dirpath, "models", "hyper_lora.pth")
        if not os.path.exists(lora_path):
            print(f"Skip {dirname} - hyper_lora.pth not found")
            continue
        if len(os.listdir(os.path.join(exp_dirpath, "images"))) >= len(df):
            print(f"Skip {dirname} - already processed")
            continue
        print(f"Processing experiment: {dirname}.", flush=True)
        print("images", len(os.listdir(os.path.join(exp_dirpath, "images"))), flush=True)

        # Load and prepare models
        model_orig = load_model_from_config(args.config, args.ckpt, args.device)
        model = load_model_from_config(args.config, args.ckpt, args.device)

        # Apply HyperLoRA to model
        lora_state_dict = torch.load(lora_path, map_location="cuda")
        hyper_lora_factory = partial(
            HyperLoRALinear,
            clip_size=768,
            rank=1,
            alpha=0.00001,
        )
        model.hyper = HypernetworkManager()
        hyper_lora_layers = inject_hyper_lora(
            model.model.diffusion_model, ["attn2.to_k", "attn2.to_v"], hyper_lora_factory
        )
        for layer_name, layer in hyper_lora_layers:
            layer.set_parent_model(model)
            model.hyper.add_hyperlora(layer_name, layer.hyper_lora)

        # Load HyperLoRA weights
        missing, unexpected = model.model.diffusion_model.load_state_dict(lora_state_dict, strict=False)
        print(f"[HyperLoRA] Loaded weights - Missing: {len(missing)}, Unexpected: {len(unexpected)}")
        print(missing)

        sampler_orig = DDIMSampler(model_orig)

        # Precompute limits
        og_num = round((args.t_enc / args.ddim_steps) * 1000)
        og_num_lim = round(((args.t_enc + 1) / args.ddim_steps) * 1000)

        # Iterate over prompts
        
        for image_id, row in df.iterrows():
            image_path = os.path.join(exp_dirpath, "images", f"{image_id:05d}.jpg")
            if os.path.exists(image_path):
                continue  # Skip if image already exists
            
            if image_id % WORLD_SIZE != RANK:
                continue
            
            prompt = row.get("prompt", "")
            if not isinstance(prompt, str) or not prompt.strip():
                print(f"Skip [{image_id}] empty prompt")
                continue
            start = time.time()
            seed = int(row.get("evaluation_seed", image_id))
            guidance = float(row.get("evaluation_guidance", 7.5))
            set_seed(seed)
            gen = torch.Generator(device=args.device).manual_seed(seed)

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

            model.hyper.set_context(t_prompt, torch.tensor([150]).to(model.device))
            model.hyper.compute_and_cache_loras(t_prompt, torch.tensor([150]).to(model.device))


            sampler = DDIMSampler(model=model)
            img = generate_images(
                sampler=sampler, model=model,
                start_code=start_code, prompt=prompt, device=model.device,
                steps=args.ddim_steps
            )

            img_np = img[0].cpu().permute(1, 2, 0).numpy()
            img_pil = to_pil_image((img_np * 255).astype(np.uint8))

            img_pil.save(image_path)
            end = time.time()
            print(f"Prompt [{prompt}] processed in {end - start:.2f}) seconds. Saved to {image_path}", flush=True)
