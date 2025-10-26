#!/usr/bin/env python3
import argparse
from pathlib import Path

import torch
from torchvision.transforms.functional import to_pil_image
from tqdm import tqdm

from ldm.models.diffusion.ddimcopy import DDIMSampler
from utils import load_model_from_config  # loads the SD model from config+ckpt


CIFAR10 = [
    "airplane","automobile","bird","cat","deer",
    "dog","frog","horse","ship","truck"
]

CIFAR100 = [
    'apple','aquarium fish','baby','bear','beaver','bed','bee','beetle','bicycle','bottle',
    'bowl','boy','bridge','bus','butterfly','camel','can','castle','caterpillar','cattle',
    'chair','chimpanzee','clock','cloud','cockroach','couch','crab','crocodile','cup','dinosaur',
    'dolphin','elephant','flatfish','forest','fox','girl','hamster','house','kangaroo','keyboard',
    'lamp','lawn mower','leopard','lion','lizard','lobster','man','maple tree','motorcycle','mountain',
    'mouse','mushroom','oak tree','orange','orchid','otter','palm tree','pear','pickup truck','pine tree',
    'plain','plate','poppy','porcupine','possum','rabbit','raccoon','ray','road','rocket',
    'rose','sea','seal','shark','shrew','skunk','skyscraper','snail','snake','spider',
    'squirrel','streetcar','sunflower','sweet pepper','table','tank','telephone','television','tiger','tractor',
    'train','trout','tulip','turtle','wardrobe','whale','willow tree','wolf','woman','worm'
]


def parse_args():
    ap = argparse.ArgumentParser("Generate CIFAR10/100 class images with SD (with tqdm)")
    ap.add_argument("--config", type=str, default="./configs/stable-diffusion/v1-inference.yaml")
    ap.add_argument("--ckpt",   type=str, default="./models/sd-v1-4.ckpt")
    ap.add_argument("--out_dir", type=str, default="samples_cifar")
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--guidance", type=float, default=7.5)   # CFG scale
    ap.add_argument("--eta", type=float, default=0.0)        # DDIM eta
    ap.add_argument("--image_size", type=int, default=512)   # assumes square
    ap.add_argument("--seed", type=int, default=2024)
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--per_class", type=int, default=20, help="Images per class")
    ap.add_argument("--batch", type=int, default=4, help="Batch size per sampling call")
    ap.add_argument("--which", type=str, default="both", choices=["cifar10","cifar100","both"])
    return ap.parse_args()


@torch.no_grad()
def generate_batch(model, sampler, prompts, device, steps=50, eta=0.0, guidance=7.5, image_size=512, seeds=None):
    """
    prompts: list[str] length = B
    seeds: list[int] or None; if given, length must be B
    returns FloatTensor [B,3,H,W] in [0,1]
    """
    B = len(prompts)
    H = W = image_size
    latent_h, latent_w = H // 8, W // 8

    if seeds is None:
        x_T = torch.randn(B, 4, latent_h, latent_w, device=device)
    else:
        xs = []
        for s in seeds:
            g = torch.Generator(device=device).manual_seed(int(s))
            xs.append(torch.randn(1, 4, latent_h, latent_w, generator=g, device=device))
        x_T = torch.cat(xs, dim=0)

    model.eval()
    with torch.autocast(device_type="cuda", enabled=("cuda" in str(device))):
        cond   = model.get_learned_conditioning(prompts)
        uncond = model.get_learned_conditioning([""] * B)

        samples_latent, _ = sampler.sample(
            S=steps,
            conditioning={"c_crossattn": [cond]},
            batch_size=B,
            shape=(4, latent_h, latent_w),
            verbose=False,
            unconditional_guidance_scale=guidance,
            unconditional_conditioning={"c_crossattn": [uncond]},
            eta=eta,
            x_T=x_T,
        )

        imgs = model.decode_first_stage(samples_latent)  # [-1,1]
        imgs = (imgs.clamp(-1, 1) + 1) / 2.0            # [0,1]
        return imgs


def save_images(imgs, out_dir: Path, start_idx: int = 0):
    out_dir.mkdir(parents=True, exist_ok=True)
    for i, im in enumerate(imgs.cpu()):
        im_u8 = (im.clamp(0, 1) * 255).round().to(torch.uint8)  # [3,H,W]
        to_pil_image(im_u8).save(out_dir / f"{start_idx + i:05d}.png")


def run_with_progress(model, sampler, class_list, root_dir: Path, args, pbar: tqdm):
    device = args.device
    per_class = args.per_class
    batch = max(1, args.batch)
    steps = args.steps
    eta = args.eta
    guidance = args.guidance
    H = args.image_size

    for cls in class_list:
        sub = root_dir / cls.replace(" ", "_")
        produced = 0
        while produced < per_class:
            bsz = min(batch, per_class - produced)
            prompts = [f"a photo of the {cls}"] * bsz
            seeds = [args.seed + produced + j for j in range(bsz)]
            pbar.set_postfix_str(f"class={cls}")
            imgs = generate_batch(
                model, sampler, prompts, device=device, steps=steps, eta=eta,
                guidance=guidance, image_size=H, seeds=seeds
            )
            save_images(imgs, sub, start_idx=produced)
            produced += bsz
            pbar.update(bsz)  # advance by number of images just saved


def main():
    args = parse_args()
    device = torch.device(args.device if (torch.cuda.is_available() or "cuda" not in args.device) else "cpu")

    # Load SD model + sampler
    model = load_model_from_config(args.config, args.ckpt, device=device)
    sampler = DDIMSampler(model=model)

    # Prepare class lists and total work
    class_lists = []
    out_roots = []
    out_root = Path(args.out_dir)

    if args.which in ("cifar100", "both"):
        class_lists.append(CIFAR100)
        out_roots.append(out_root / "cifar100")
    if args.which in ("cifar10", "both"):
        class_lists.append(CIFAR10)
        out_roots.append(out_root / "cifar10")

    total_images = sum(len(cl) for cl in class_lists) * args.per_class

    # Single global tqdm bar
    with tqdm(total=total_images, unit="img", desc="Generating") as pbar:
        for cls_list, root in zip(class_lists, out_roots):
            run_with_progress(model, sampler, cls_list, root, args, pbar)

    print(f"Done. Images saved under: {out_root}")


if __name__ == "__main__":
    main()
