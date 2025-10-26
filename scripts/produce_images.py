#!/usr/bin/env python3
import argparse
from pathlib import Path

import torch
from torchvision.transforms.functional import to_pil_image

from ldm.models.diffusion.ddimcopy import DDIMSampler
from ldm.util import instantiate_from_config
from utils import load_model_from_config  # this should load the SD model from config+ckpt


def parse_args():
    ap = argparse.ArgumentParser("Minimal SD image generation")
    ap.add_argument("--config", type=str, default="./configs/stable-diffusion/v1-inference.yaml")
    ap.add_argument("--ckpt",   type=str, default="./models/sd-v1-4.ckpt")
    ap.add_argument("--prompt", type=str, default="a photo of a ship")
    ap.add_argument("--out_dir", type=str, default="samples_minimal")
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--guidance", type=float, default=7.5)   # CFG scale
    ap.add_argument("--eta", type=float, default=0.0)        # DDIM eta
    ap.add_argument("--samples", type=int, default=4)        # number of images
    ap.add_argument("--image_size", type=int, default=512)   # assumes square
    ap.add_argument("--seed", type=int, default=2024)
    ap.add_argument("--device", type=str, default="cuda:0")
    return ap.parse_args()


@torch.no_grad()
def generate_images(model, sampler, prompt, device, steps=50, eta=0.0, n=1, start_seed=0, image_size=512, guidance=7.5, out_dir="samples_minimal"):
    """
    Generate n images with classifier-free guidance from CompVis SD model.
    Saves PNGs and returns a [n,3,H,W] tensor in [0,1].
    """
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    H = W = image_size
    latent_h, latent_w = H // 8, W // 8

    # batch all 'n' at once for speed (if VRAM allows)
    gen = torch.Generator(device=device).manual_seed(start_seed)
    x_T = torch.randn(n, 4, latent_h, latent_w, generator=gen, device=device)

    model.eval()
    with torch.autocast(device_type="cuda", enabled=("cuda" in device)):
        cond   = model.get_learned_conditioning([prompt] * n)
        uncond = model.get_learned_conditioning([""] * n)

        samples_latent, _ = sampler.sample(
            S=steps,
            conditioning={"c_crossattn": [cond]},
            batch_size=n,
            shape=(4, latent_h, latent_w),
            verbose=False,
            unconditional_guidance_scale=guidance,
            unconditional_conditioning={"c_crossattn": [uncond]},
            eta=eta,
            x_T=x_T,
        )

        imgs = model.decode_first_stage(samples_latent)            # [-1,1]
        imgs = (imgs.clamp(-1, 1) + 1) / 2.0                       # [0,1]

    # save
    for i, im in enumerate(imgs.cpu()):
        (to_pil_image((im.clamp(0, 1) * 255).round().to(torch.uint8))
         .save(Path(out_dir) / f"sample_{i:04d}.png"))

    return imgs


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() or "cuda" not in args.device else "cpu")

    # Load model
    model = load_model_from_config(args.config, args.ckpt, device=device)
    sampler = DDIMSampler(model=model)

    # Generate
    _ = generate_images(
        model, sampler,
        prompt=args.prompt,
        device=str(device),
        steps=args.steps,
        eta=args.eta,
        n=args.samples,
        start_seed=args.seed,
        image_size=args.image_size,
        guidance=args.guidance,
        out_dir=args.out_dir,
    )
    print(f"Saved {args.samples} image(s) to: {args.out_dir}")


if __name__ == "__main__":
    main()
