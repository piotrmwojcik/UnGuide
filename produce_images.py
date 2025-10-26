#!/usr/bin/env python3
import argparse
from pathlib import Path

import torch
from torchvision.transforms.functional import to_pil_image
from tqdm import tqdm
from accelerate import Accelerator

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
    ap = argparse.ArgumentParser("Distributed SD generation for CIFAR10/100 (Accelerate)")
    ap.add_argument("--config", type=str, default="./configs/stable-diffusion/v1-inference.yaml")
    ap.add_argument("--ckpt",   type=str, default="./models/sd-v1-4.ckpt")
    ap.add_argument("--out_dir", type=str, default="samples_cifar")
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--guidance", type=float, default=7.5)   # CFG scale
    ap.add_argument("--eta", type=float, default=0.0)        # DDIM eta
    ap.add_argument("--image_size", type=int, default=512)   # assumes square
    ap.add_argument("--seed", type=int, default=2024)
    ap.add_argument("--per_class", type=int, default=20, help="Images per class")
    ap.add_argument("--batch", type=int, default=4, help="Batch size per sampling call (per process)")
    ap.add_argument("--which", type=str, default="both", choices=["cifar10","cifar100","both"])
    return ap.parse_args()


@torch.no_grad()
def generate_batch(model, sampler, prompts, device, steps=50, eta=0.0, guidance=7.5, image_size=512, seeds=None, use_amp=True):
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
    with torch.autocast(device_type="cuda", enabled=(use_amp and device.type == "cuda")):
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


def partition_classes(classes, rank, world):
    """Disjoint slice of classes for this process."""
    return classes[rank::world]


def run_partition(model, sampler, class_list, root_dir: Path, args, accelerator: Accelerator, pbar: tqdm = None):
    device = accelerator.device
    per_class = args.per_class
    batch = max(1, args.batch)
    steps = args.steps
    eta = args.eta
    guidance = args.guidance
    H = args.image_size
    use_amp = (accelerator.mixed_precision != "no")

    # Distinct seed stream per rank to avoid duplicates
    base_seed = int(args.seed) + accelerator.process_index * 1_000_000

    for cls in class_list:
        sub = root_dir / cls.replace(" ", "_")
        produced = 0
        while produced < per_class:
            bsz = min(batch, per_class - produced)
            prompts = [f"a photo of the {cls}"] * bsz
            seeds = [base_seed + produced + j for j in range(bsz)]
            if accelerator.is_main_process and pbar is not None:
                pbar.set_postfix_str(f"class={cls}")
            imgs = generate_batch(
                model, sampler, prompts,
                device=device, steps=steps, eta=eta,
                guidance=guidance, image_size=H, seeds=seeds, use_amp=use_amp
            )
            save_images(imgs, sub, start_idx=produced)
            produced += bsz
            if accelerator.is_main_process and pbar is not None:
                pbar.update(bsz)  # only rank-0 updates its local bar


def main():
    args = parse_args()
    accelerator = Accelerator()  # uses env / accelerate config
    device = accelerator.device

    # Load SD model + sampler on this process's device
    model = load_model_from_config(args.config, args.ckpt, device=device)
    sampler = DDIMSampler(model=model)

    out_root = Path(args.out_dir)

    # Build global class lists, then take a per-rank partition to avoid clashes
    selected_sets = []
    if args.which in ("cifar100", "both"):
        selected_sets.append(("cifar100", CIFAR100))
    if args.which in ("cifar10", "both"):
        selected_sets.append(("cifar10", CIFAR10))

    # Rank-0 progress bar over its OWN workload (simple & safe)
    total_rank0 = 0
    if accelerator.is_main_process:
        for _, clist in selected_sets:
            total_rank0 += len(partition_classes(clist, 0, accelerator.num_processes)) * args.per_class
        pbar = tqdm(total=total_rank0, unit="img", desc="Generating (rank0)")
    else:
        pbar = None

    # Run each selected set
    for subset_name, full_list in selected_sets:
        part = partition_classes(full_list, accelerator.process_index, accelerator.num_processes)
        root_dir = out_root / subset_name
        if part:
            run_partition(model, sampler, part, root_dir, args, accelerator, pbar)

    if accelerator.is_main_process and pbar is not None:
        pbar.close()
        print(f"Done. Images saved under: {out_root}")

if __name__ == "__main__":
    main()
