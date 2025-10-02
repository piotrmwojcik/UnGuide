import argparse
import json
import os
import torch
import torch
import math
import numpy as np
from transformers import CLIPTextModel, CLIPTokenizer
from tqdm import tqdm
from functools import partial
import time

from utils import set_seed, load_model_from_config, apply_lora_to_model
from sampling import sample_model
from hyper_lora import (HyperLoRALinear, inject_hyper_lora,
                        inject_hyper_lora_nsfw)
from ldm.models.diffusion.ddimcopy import DDIMSampler


def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyze latent differences between original and LoRA-augmented Stable Diffusion models"
    )
    parser.add_argument(
        "--config", required=True,
        help="Path to the model config YAML file"
    )
    parser.add_argument(
        "--ckpt", required=True,
        help="Path to the model checkpoint (.ckpt)"
    )
    parser.add_argument(
        "--lora", required=True,
        help="Path to the LoRA state dict (.pth)"
    )
    parser.add_argument(
        "--prompts_json", required=True,
        help="Path to JSON file containing an array of prompts"
    )
    parser.add_argument(
        "--output_dir", required=True,
        help="Directory where results will be saved"
    )
    parser.add_argument(
        "--device", default="cuda:0",
        help="Device to run the models on"
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--ddim_steps", type=int, default=50,
        help="Number of DDIM sampling steps"
    )
    parser.add_argument(
        "--image_size", type=int, default=512,
        help="Size of the generated image (square)"
    )
    parser.add_argument(
        "--guidance", type=float, default=7.5,
        help="Classifier-free guidance scale"
    )
    parser.add_argument(
        "--repeats", type=int, default=1,
        help="Total number of sampling repeats"
    )
    parser.add_argument(
        "--eta", type=float, default=0.0,
        help="DDIM eta (noise schedule parameter)"
    )
    parser.add_argument(
        "--t_enc", type=int, default=25,
        help="Encoding timestep at which to compare model outputs"
    )
    parser.add_argument(
        "--n_samples", type=int, default=25,
        help="n_samples"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Optionally set the random seed for reproducibility
    if args.seed is not None:
        set_seed(args.seed)

    # Load original and LoRA-augmented models
    model = load_model_from_config(args.config, args.ckpt, args.device)
    model_orig = load_model_from_config(args.config, args.ckpt, args.device)

    # Load LoRA parameters and apply to the model
    lora_sd = torch.load(args.lora, map_location="cpu")
    #apply_lora_to_model(model.model.diffusion_model, lora_sd, alpha=8)

    hyper_lora_factory = partial(
        HyperLoRALinear,
        clip_size=768,
        rank=1,
        alpha=0.001,
    )
    hyper_lora_layers = inject_hyper_lora(
        model.model.diffusion_model, ["attn2.to_k", "attn2.to_v"], hyper_lora_factory
    )

    for layer in hyper_lora_layers:
        layer.set_parent_model(model)

    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
    clip_text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14").to(args.device).eval()

    updated = 0
    skipped = []

    sd = model.model.diffusion_model.state_dict()

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

    # Initialize DDIM samplers
    sampler_orig = DDIMSampler(model_orig)

    # Compute numeric bounds for random timesteps based on t_enc
    og_num = round((args.t_enc / args.ddim_steps) * 1000)
    og_num_lim = round(((args.t_enc + 1) / args.ddim_steps) * 1000)

    # Load prompts from JSON file
    with open(args.prompts_json, 'r', encoding='utf-8') as f:
        data = json.load(f)
    prompts = [data.get("target")] + data.get("other", []) + data.get("synonyms", [])
    # Prepare storage for latent difference norms
    prompt_diffs = {prompt: [] for prompt in prompts}

    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)

    with torch.no_grad():
        total = args.repeats
        for sample_idx in range(0, total, args.n_samples):
            start = time.time()
            n_samples = args.n_samples

            # Initialize random latent codes and timesteps
            start_codes = torch.randn(
                (n_samples, 4, args.image_size // 8, args.image_size // 8),
                device=args.device
            )
            t_enc_ddpm = torch.randint(
                og_num, og_num_lim,
                (n_samples,), device=args.device
            )

            for prompt in tqdm(prompts, desc="Processing prompts"):
                # Get conditioning embeddings
                cond = model.get_learned_conditioning([prompt] * n_samples)
                cond_orig = model_orig.get_learned_conditioning([prompt] * n_samples)

                # Sample noisy latents up to t_enc
                z_batch = sample_model(
                    model_orig,
                    sampler_orig,
                    cond_orig,
                    args.image_size,
                    args.image_size,
                    args.ddim_steps,
                    args.guidance,
                    args.eta,
                    start_code=start_codes,
                    n_samples=n_samples,
                    till_T=args.t_enc,
                    verbose=False,
                )

                print(prompts[0])
                inputs = tokenizer(
                    prompts[0],
                    max_length=tokenizer.model_max_length,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt",
                ).to(args.device).input_ids

                t_prompt = clip_text_encoder(inputs).pooler_output.detach()


                # Compute epsilon predictions for both models
                model.current_conditioning = t_prompt
                eps_lora = model.apply_model(z_batch, t_enc_ddpm, cond)
                #model.current_conditioning = cond_orig
                eps_orig = model_orig.apply_model(z_batch, t_enc_ddpm, cond_orig)

                # Compute norm of the difference and record it
                diffs = (
                    eps_lora - eps_orig
                ).view(n_samples, -1).norm(dim=1).cpu().numpy().tolist()
                prompt_diffs[prompt].extend(diffs)

                # Free up GPU memory
                torch.cuda.empty_cache()
            end = time.time()
            print(f"Sample: {sample_idx} Time: {end - start}", flush=True)

    # Calculate and print average differences per prompt
    prompt_avgs = [
        (prompt, np.mean(vals))
        for prompt, vals in prompt_diffs.items()
    ]
    prompt_avgs.sort(key=lambda x: x[1], reverse=True)
    prompt_avgs = dict(prompt_avgs)
    out = {
        "prompt_avgs": prompt_avgs,
        "config": {
            "config": args.config,
            "ckpt": args.ckpt,
            "lora": args.lora,
            "device": args.device,
            "seed": args.seed,
            "ddim_steps": args.ddim_steps,
            "image_size": args.image_size,
            "guidance": args.guidance,
            "repeats": args.repeats,
            "eta": args.eta,
            "t_enc": args.t_enc,
            "n_samples": args.n_samples,
            "target_prompt": data.get("target"),
            "synonyms": data.get("synonyms"),
        }
    }
    for prompt, avg in prompt_avgs.items():
        print(f"Prompt: {prompt}, Mean latent difference: {avg:.4f}")

    # Save full dictionary of differences
    output_path = os.path.join(args.output_dir, f"calc_diff_results.json")
    with open(output_path, 'w', encoding='utf-8') as fout:
        json.dump(out, fout, ensure_ascii=False, indent=2)
    print(f"Saved latent differences to {output_path}")


if __name__ == "__main__":
    main()
