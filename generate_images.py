import os
import json
import argparse
import torch
import torch.nn as nn
from train import create_quick_sampler, _iter_hyperlora_layers
from functools import partial
from transformers import CLIPTextModel, CLIPTokenizer
from ldm.models.diffusion.ddimcopy import DDIMSampler
from utils import load_model_from_config, apply_lora_to_model, set_seed
from torchvision.transforms.functional import to_pil_image
from autoguide import AutoGuidedModel
from hyper_lora import (HyperLoRALinear, inject_hyper_lora,
                        inject_hyper_lora_nsfw)
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
        "--ckpt", type=str, default="models/sd-v1-4-full-ema.ckpt",
        help="path to model checkpoint"
    )
    # parser.add_argument(
    #     "--lora", type=str, default="original_train.pth",
    #     help="path to LoRA state dict"
    # )
    # parser.add_argument(
    #     "--train_json", type=str, default="data/train_json.json",
    #     help="prompts for image generation"
    # )
    # parser.add_argument(
    #     "--diff_results", type=str, default="calc_diff_results.json",
    #     help="path to JSON file with difference results"
    # )
    parser.add_argument(
        "--output_dir", type=str, default="cat",
        help=""
    )
    parser.add_argument(
        "--samples", type=int, default=50,
        help="number of images to generate"
    )
    parser.add_argument(
        "--w1", type=float, default=-1.0,
        help="weight for prompt above threshold"
    )
    parser.add_argument(
        "--w2", type=float, default=2.0,
        help="weight for prompt below threshold"
    )
    parser.add_argument(
        "--steps", type=int, default=50,
        help="number of sampling steps"
    )
    parser.add_argument("--start_guidance", type=float, default=9.0,
                        help="Starting guidance scale")
    parser.add_argument("--image_size", type=int, default=512,
                        help="Image size for training")
    parser.add_argument("--ddim_steps", type=int, default=50,
                        help="DDIM sampling steps")
    parser.add_argument("--ddim_eta", type=float, default=0.0,
                        help="DDIM eta")
    parser.add_argument(
        "--seed", type=int, default=2024,
        help="random seed for reproducibility"
    )
    parser.add_argument(
        "--device", type=str, default="cuda:0",
        help="device to run generation on"
    )

    return parser.parse_args()


def decide_w(prompt, prompt_empty, w1=-1, w2=2):
    return w1 if prompt > prompt_empty else w2


def generate_image(
        sampler, auto_model, start_code, cond, uncond, steps
):
    with torch.no_grad():
        samples, _ = sampler.sample(
            S=steps,
            conditioning=cond,
            unconditional_conditioning=uncond,
            batch_size=start_code.shape[0],
            shape=start_code.shape[1:],
            verbose=False,
            eta=0.0,
            x_T=start_code,
            mode="auto",
        )
        decoded = auto_model.decode_first_stage(samples)
        decoded = (decoded + 1.0) / 2.0
        decoded = torch.clamp(decoded, 0.0, 1.0)
    return decoded


def flatten_live_tensors(model: nn.Module) -> torch.Tensor:
    """
m    This preserves autograd so loss can backprop through them.
    """
    parts: List[torch.Tensor] = []

    _LIVE_GETTERS = [
        ("x_L", lambda hl: getattr(hl, "_last_x_L", None)),
    ]

    for _, hl in _iter_hyperlora_layers(model):
        for _, getter in _LIVE_GETTERS:
            t = getter(hl)
            if t is None:
                continue
            # ensure Tensor, keep graph
            parts.append(t.view(-1))

    if not parts:
        return torch.tensor([], device=next(model.parameters()).device)

    return torch.cat(parts, dim=0)


if __name__ == "__main__":
    RANK = int(os.environ.get("RANK", "0"))
    WORLD_SIZE = int(os.environ.get("WORLD_SIZE", "1"))

    args = parse_args()

    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
    clip_text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14").to(args.device).eval()


    exps = os.listdir(args.output_dir)
    print(f"Exps: {exps}", flush=True)
    for exp in exps:
        exp_filepath = os.path.join(args.output_dir, exp)
        img_root = os.path.join(args.output_dir, exp, "images")
        lora_filepath = os.path.join(exp_filepath, "models", "hyper_lora.pth")

        #diff_results_path = os.path.join(exp_filepath, "calc_diff_results.json")
        train_json_path = os.path.join(exp_filepath, "train_config.json")

        #with open(diff_results_path, 'r') as f:
        #    results = json.load(f)

        with open(train_json_path, 'r') as f:
            settings = json.load(f)

        prompts_json_path = settings["prompts_json"]
        with open(prompts_json_path, "r") as f:
            data = json.load(f)

        prompts = [data.get("target")] + data.get("synonyms", []) + data.get("other", [])
        prompts = prompts[:-1]
        print("Prompts: ", prompts, flush=True)

        # collect all valid subfolders
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
            print(f"Skip: {exp}", flush=True)
            continue

        print(f"Exp: {exp}", flush=True)
        # Load models
        model_full = load_model_from_config(
            args.config, args.ckpt, device=args.device
        )
        model_unl = load_model_from_config(
            args.config, args.ckpt, device=args.device
        )

        # Apply LoRA to unlearned model
        lora_sd = torch.load(lora_filepath, map_location=args.device)
        hyper_lora_factory = partial(
            HyperLoRALinear,
            clip_size=768,
            rank=1,
            alpha=0.00001,
        )
        hyper_lora_layers = inject_hyper_lora(
            model_unl.model.diffusion_model, ["attn2.to_k", "attn2.to_v"], hyper_lora_factory
        )
        for layer in hyper_lora_layers:
            layer.set_parent_model(model_unl)

        updated = 0
        skipped = []

        sd = model_unl.model.diffusion_model.state_dict()

        with torch.no_grad():
            for k, v in lora_sd.items():
                if k in sd:
                    if torch.is_tensor(lora_sd[k]) and torch.is_tensor(v) and lora_sd[k].shape == v.shape:
                        # match dtype/device of target param/buffer
                        sd[k].copy_(v.to(sd[k].dtype))
                        updated += 1
                        print("updated:", k)
                    else:
                        skipped.append((k, "shape/dtype mismatch"))
                else:
                    skipped.append((k, "no such key in model"))

        print(f"[LoRA] copied {updated} tensors, skipped {len(skipped)}")

        for prompt in prompts:
            class_name = prompt.split(" ")[-1]
            class_root = os.path.join(img_root, class_name)
            os.makedirs(class_root, exist_ok=True)
            if len(os.listdir(class_root)) == args.samples:
                continue

            # Conditioning
            cond = model_unl.get_learned_conditioning([prompt])
            uncond = model_unl.get_learned_conditioning([""])

            sampler_unl = DDIMSampler(model=model_unl)
            quick_sampler = create_quick_sampler(model_unl, sampler_unl,
                                                 args.image_size, args.ddim_steps, args.ddim_eta)

            t_enc = torch.randint(args.ddim_steps, (1,), device=model_unl.device)
            og_num = round((int(t_enc) / args.ddim_steps) * 1000)
            og_num_lim = round((int(t_enc + 1) / args.ddim_steps) * 1000)
            t_enc_ddpm = torch.randint(og_num, og_num_lim, (1,), device=model_unl.device)

            start_code = torch.randn(
                (1, 4, args.image_size // 8, args.image_size // 8),
                device=model_unl.device
            )
            inputs = tokenizer(
                prompt,
                max_length=tokenizer.model_max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            ).to(args.device).input_ids

            inputs_empty = tokenizer(
                "",
                max_length=tokenizer.model_max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            ).to(args.device).input_ids

            t_prompt = clip_text_encoder(inputs).pooler_output.detach()
            empty_prompt = clip_text_encoder(inputs_empty).pooler_output.detach()

            model_unl.current_conditioning = t_prompt
            model_unl.time_step = 150

            z = quick_sampler(cond, args.start_guidance, start_code, int(t_enc))
            _ = model_unl.apply_model(z, t_enc_ddpm, cond)
            tensors_flat_t_live_t1 = flatten_live_tensors(model_unl)

            #z = quick_sampler(uncond, args.start_guidance, start_code, int(t_enc))
            #model_unl.current_conditioning = empty_prompt
            model_unl.time_step = 0
            _ = model_unl.apply_model(z, t_enc_ddpm, uncond)
            tensors_flat_t_live_t0 = flatten_live_tensors(model_unl)
            tensors_flat_t_live = tensors_flat_t_live_t1 - tensors_flat_t_live_t0
            with torch.no_grad():
                if isinstance(tensors_flat_t_live, torch.Tensor):
                    vec = tensors_flat_t_live.reshape(-1).float()
                else:
                    vec = torch.cat([t.reshape(-1).float() for t in tensors_flat_t_live], dim=0)
                l2 = vec.norm(p=2).item()

            print("prompt: ", prompt, f"||tensors_flat_t_live||_2 = {l2:.6f}")

            if l2 < 1.2:
                model = model_unl
            else:
                model = model_full
            #w = -1
            #w = decide_w(
            #    results["prompt_avgs"].get(prompt), results["prompt_avgs"].get(""),
            #    w1=args.w1, w2=args.w2
            #)
            #w = 0
            # Prepare models and sampler
            #auto_model = AutoGuidedModel(
            #    model_full, model_unl, w=w
            #).eval()
            sampler = DDIMSampler(model=model)

            print(f"cond dimensions {cond.size()}")
            print(f"uncond dimensions {uncond.size()}")
            # Generation loop

            for idx in tqdm(range(args.samples), desc="Generating images"):
                start = time.time()
                filename = f"{idx:05d}.jpg"
                filename_path = os.path.join(img_root, class_name, filename)
                if os.path.exists(filename_path):
                    continue
                if idx % WORLD_SIZE != RANK:
                    continue

                seed = args.seed + idx
                set_seed(seed)
                gen = torch.Generator(device=args.device).manual_seed(seed)

                start_code = torch.randn(1, 4, 64, 64, generator=gen, device=args.device)
                inputs = tokenizer(
                    prompt,
                    max_length=tokenizer.model_max_length,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt",
                ).to(args.device).input_ids

                t_prompt = clip_text_encoder(inputs).pooler_output.detach()

                model_unl.current_conditioning = t_prompt
                model_unl.time_step = 150

                img = generate_image(
                    sampler, model, start_code, cond, uncond, args.steps
                )
                img_np = img[0].cpu().permute(1, 2, 0).numpy()
                img_pil = to_pil_image((img_np * 255).astype(np.uint8))

                img_pil.save(filename_path, format='JPEG', quality=90, optimize=True)
                end = time.time()
                print(f"Generate: {end - start}", flush=True)
