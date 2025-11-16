import os
import json
import argparse
import torch
from functools import partial
from transformers import CLIPTextModel, CLIPTokenizer
from hyper_lora import HyperLoRALinear, HypernetworkManager, inject_hyper_lora
from ldm.models.diffusion.ddimcopy import DDIMSampler
from utils import load_model_from_config, set_seed
from torchvision.transforms.functional import to_pil_image
import numpy as np
import time
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(
        description="Auto-guided image generation with Stable Diffusion and LoRA")
    parser.add_argument(
        "--config", type=str, default="./configs/stable-diffusion/v1-inference.yaml",
        help="path to model config file"
    )
    parser.add_argument(
        "--ckpt", type=str, default="models/sd-v1-4.ckpt",
        help="path to model checkpoint"
    )
    parser.add_argument(
        "--output_dir", type=str, required=True,
        help="directory containing experiment folders with LoRA_fusion_model/hyper_lora.pth"
    )
    parser.add_argument(
        "--class_name", type=str, required=True,
        help="class name to generate (e.g., 'airplane', 'cat', 'dog'). Will load data/{class_name}.json"
    )
    parser.add_argument(
        "--save_folder", type=str, default="images",
        help="subfolder name to save generated images"
    )
    parser.add_argument(
        "--samples", type=int, default=50,
        help="number of images to generate per prompt"
    )
    parser.add_argument(
        "--steps", type=int, default=50,
        help="number of sampling steps"
    )
    parser.add_argument("--guidance_scale", type=float, default=7.5,
                        help="CFG guidance scale")
    parser.add_argument("--image_size", type=int, default=512,
                        help="Image size for generation")
    parser.add_argument("--ddim_steps", type=int, default=50,
                        help="DDIM sampling steps")
    parser.add_argument("--ddim_eta", type=float, default=0.0,
                        help="DDIM eta")
    parser.add_argument("--hyper_timestep", type=int, default=500,
                        help="Timestep for HyperLoRA context")
    parser.add_argument(
        "--alpha", type=float, default=0.00001,
        help="LoRA alpha scaling factor"
    )
    parser.add_argument(
        "--seed", type=int, default=2024,
        help="random seed base for reproducibility"
    )
    parser.add_argument(
        "--device", type=str, default="cuda:0",
        help="device to run generation on"
    )

    return parser.parse_args()


class CombinedCFGModel:
    """Wrapper that uses different models for conditional and unconditional passes."""

    def __init__(self, cond_model, uncond_model):
        self.cond_model = cond_model
        self.uncond_model = uncond_model
        self.device = cond_model.device

    def apply_model(self, x, t, c):
        # When DDIMSampler uses guidance, it concatenates [uncond, cond] inputs
        # We split and route to different models
        b2 = x.shape[0]
        assert b2 % 2 == 0
        b = b2 // 2

        x_uncond = x[:b]
        x_cond = x[b:]
        t_uncond = t[:b]
        t_cond = t[b:]

        # Split conditioning
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

        # Route unconditional to model_full, conditional to model_unl
        out_uncond = self.uncond_model.apply_model(x_uncond, t_uncond, c_uncond)
        out_cond = self.cond_model.apply_model(x_cond, t_cond, c_cond)

        return torch.cat([out_uncond, out_cond], dim=0)

    def get_learned_conditioning(self, prompts):
        return self.cond_model.get_learned_conditioning(prompts)

    def decode_first_stage(self, z):
        return self.cond_model.decode_first_stage(z)

    def eval(self):
        self.cond_model.eval()
        self.uncond_model.eval()
        return self

    def __getattr__(self, name):
        return getattr(self.cond_model, name)


def generate_images(
    sampler,
    model,
    prompt: str,
    device: torch.device,
    steps: int = 50,
    eta: float = 0.0,
    batch_size: int = 1,
    start_code: torch.Tensor = None,
    guidance_scale: float = 7.5,
):
    if start_code is None:
        start_code = torch.randn(batch_size, 4, 64, 64, device=device)

    model.eval()
    with torch.no_grad(), torch.autocast(device_type=device.type, enabled=(device.type == "cuda")):
        cond = model.get_learned_conditioning([prompt] * start_code.shape[0])
        uncond = model.get_learned_conditioning([""] * start_code.shape[0])

        samples, _ = sampler.sample(
            S=steps,
            conditioning={"c_crossattn": [cond]},
            batch_size=start_code.shape[0],
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


if __name__ == "__main__":
    LOCAL_RANK = int(os.environ.get("LOCAL_RANK", "0"))
    RANK = int(os.environ.get("RANK", 0))
    WORLD_SIZE = int(os.environ.get("WORLD_SIZE", 1))

    torch.cuda.set_device(LOCAL_RANK)
    device = torch.device(f"cuda:{LOCAL_RANK}")
    args = parse_args()

    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
    clip_text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14").to(device).eval()

    # Load class-specific prompts from JSON
    json_path = f"data/{args.class_name}.json"
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"JSON file not found: {json_path}")

    with open(json_path, "r") as f:
        data = json.load(f)

    prompts = [data.get("target")] + data.get("synonyms", []) + data.get("other", [])
    prompts = prompts[:-1]
    print(f"Loaded prompts from {json_path}: {prompts}", flush=True)

    # Check if output_dir contains a single experiment or multiple experiments
    if os.path.exists(os.path.join(args.output_dir, "LoRA_fusion_model", "hyper_lora.pth")):
        dirs = [args.output_dir]
    else:
        dirs = [os.path.join(args.output_dir, d) for d in os.listdir(args.output_dir)
                if os.path.isdir(os.path.join(args.output_dir, d))]

    for exp_dirpath in dirs:
        print(f"Processing experiment: {exp_dirpath}", flush=True)

        img_root = os.path.join(exp_dirpath, args.save_folder)
        os.makedirs(img_root, exist_ok=True)
        lora_path = os.path.join(exp_dirpath, "LoRA_fusion_model", "hyper_lora.pth")

        if not os.path.exists(lora_path):
            print(f"Skip {exp_dirpath} - hyper_lora.pth not found at {lora_path}")
            continue

        # Check if already processed
        subs = [
            d for d in os.listdir(img_root)
            if os.path.isdir(os.path.join(img_root, d))
        ] if os.path.isdir(img_root) else []

        counts = []
        for sub in subs:
            path = os.path.join(img_root, sub)
            imgs = [
                f for f in os.listdir(path)
                if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))
            ]
            counts.append(len(imgs))

        if len(prompts) * args.samples == sum(counts):
            print(f"Skip {exp_dirpath} - already processed", flush=True)
            continue

        # Load models - model_orig for unconditional, model for conditional
        model_orig = load_model_from_config(args.config, args.ckpt, device)
        model = load_model_from_config(args.config, args.ckpt, device)

        # Apply HyperLoRA to conditional model only
        lora_sd = torch.load(lora_path, map_location=device)
        hyper_lora_factory = partial(
            HyperLoRALinear,
            clip_size=768,
            rank=1,
            train_steps=args.hyper_timestep,
            alpha=args.alpha,
        )
        model.hyper = HypernetworkManager()
        hyper_lora_layers = inject_hyper_lora(
            model.model.diffusion_model, ["attn2.to_k", "attn2.to_v"], hyper_lora_factory
        )
        for layer_name, layer in hyper_lora_layers:
            layer.set_parent_model(model)
            model.hyper.add_hyperlora(layer_name, layer.hyper_lora)

        updated = 0
        skipped = []

        sd = model.model.diffusion_model.state_dict()

        with torch.no_grad():
            for k, v in lora_sd.items():
                if k in sd:
                    if torch.is_tensor(lora_sd[k]) and torch.is_tensor(v) and lora_sd[k].shape == v.shape:
                        sd[k].copy_(v.to(sd[k].dtype))
                        updated += 1
                        print("updated:", k)
                    else:
                        skipped.append((k, "shape/dtype mismatch"))
                else:
                    skipped.append((k, "no such key in model"))

        print(f"[LoRA] copied {updated} tensors, skipped {len(skipped)}")

        # Generate images for each prompt
        for prompt in prompts:
            if not prompt or not prompt.strip():
                continue

            with torch.no_grad():
                class_name = prompt.split(" ")[-1]
                class_root = os.path.join(img_root, class_name)
                os.makedirs(class_root, exist_ok=True)

                if len(os.listdir(class_root)) >= args.samples:
                    print(f"Skip {prompt} - already has {args.samples} samples")
                    continue

                # Use combined model: conditional uses model (with LoRA), unconditional uses model_orig
                combined_model = CombinedCFGModel(cond_model=model, uncond_model=model_orig).eval()
                sampler = DDIMSampler(model=combined_model)

                for idx in tqdm(range(args.samples), desc=f"Generating '{prompt}'"):
                    filename = f"{idx:05d}.jpg"
                    filename_path = os.path.join(class_root, filename)
                    if os.path.exists(filename_path):
                        continue
                    if idx % WORLD_SIZE != RANK:
                        continue

                    start = time.time()
                    seed = args.seed + idx
                    set_seed(seed)
                    gen = torch.Generator(device=device).manual_seed(seed)

                    start_code = torch.randn(1, 4, args.image_size // 8, args.image_size // 8,
                                            generator=gen, device=device)
                    inputs = tokenizer(
                        prompt,
                        max_length=tokenizer.model_max_length,
                        padding="max_length",
                        truncation=True,
                        return_tensors="pt",
                    ).to(device).input_ids

                    t_prompt = clip_text_encoder(inputs).pooler_output.detach()

                    model.hyper.set_context(t_prompt, torch.tensor([args.hyper_timestep]).to(model.device))
                    model.hyper.compute_and_cache_loras(t_prompt, torch.tensor([args.hyper_timestep]).to(model.device))

                    img = generate_images(
                        sampler=sampler, model=combined_model,
                        start_code=start_code, prompt=prompt, device=model.device,
                        steps=args.steps, guidance_scale=args.guidance_scale
                    )
                    img_np = img[0].cpu().permute(1, 2, 0).numpy()
                    img_pil = to_pil_image((img_np * 255).astype(np.uint8))

                    img_pil.save(filename_path, format='JPEG', quality=90, optimize=True)
                    end = time.time()
                    print(f"Generated {filename_path} in {end - start:.2f}s", flush=True)
