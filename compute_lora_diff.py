import argparse
import json
import os
import torch
import torch
import math
import numpy as np
from tqdm import tqdm
from functools import partial
import time

from utils import set_seed, load_model_from_config, apply_lora_to_model
from sampling import sample_model
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
        clip_size=args.clip_size,
        rank=args.lora_rank,
        alpha=args.lora_alpha,
    )
    hyper_lora_layers = inject_hyper_lora(
        model.model.diffusion_model, args.target_modules, hyper_lora_factory
    )

    for layer in hyper_lora_layers:
        layer.set_parent_model(model)

    def is_leaf(m):
        return len(list(m.children())) == 0

    for name, module in model.model.diffusion_model.named_modules():
        if is_leaf(module):
            print(f"{name}: {module.__class__.__name__}")

    def report_weight_diffs(model_after, model_before, topk=20, eps=1e-12, change_thresh=0.0):
        """
        Compare state_dicts and print per-tensor diffs:
          - l2 norm of delta
          - relative l2 (||Δ|| / (||W_before|| + eps))
          - max |delta|
          - fraction of elements with |delta| > change_thresh
        """
        sd_after = model_after.state_dict()
        sd_before = model_before.state_dict()

        keys_after = set(sd_after.keys())
        keys_before = set(sd_before.keys())

        only_after = sorted(keys_after - keys_before)
        only_before = sorted(keys_before - keys_after)

        if only_after:
            print(f"[INFO] {len(only_after)} tensors exist only in AFTER (likely LoRA aux):")
            for k in only_after[:10]:
                print("   +", k)
            if len(only_after) > 10:
                print("   ...")

        if only_before:
            print(f"[INFO] {len(only_before)} tensors exist only in BEFORE (unexpected):")
            for k in only_before[:10]:
                print("   -", k)
            if len(only_before) > 10:
                print("   ...")

        shared = sorted(keys_after & keys_before)

        rows = []
        total_elems = 0
        changed_elems = 0
        total_tensors = 0
        changed_tensors = 0

        for k in shared:
            a = sd_after[k]
            b = sd_before[k]

            # focus on floating tensors only
            if not torch.is_floating_point(a) or not torch.is_floating_point(b):
                continue
            if a.shape != b.shape:
                print(f"[WARN] shape mismatch for {k}: after={tuple(a.shape)} before={tuple(b.shape)}; skipping")
                continue

            total_tensors += 1
            with torch.no_grad():
                da = (a.detach().float().cpu() - b.detach().float().cpu())
                nb = torch.linalg.vector_norm(b.detach().float().cpu())
                nd = torch.linalg.vector_norm(da)
                rel = (nd / (nb + eps)).item()
                maxabs = da.abs().max().item()
                elems = da.numel()
                nz = (da.abs() > change_thresh).sum().item()

                total_elems += elems
                changed_elems += nz
                if maxabs > 0:
                    changed_tensors += 1

                rows.append({
                    "name": k,
                    "shape": tuple(a.shape),
                    "||Δ||2": nd.item(),
                    "rel||Δ||": rel,
                    "max|Δ|": maxabs,
                    "changed_frac": nz / elems if elems > 0 else math.nan,
                    "elems": elems,
                })

        # sort by relative change, then absolute l2
        rows_sorted = sorted(rows, key=lambda r: (r["rel||Δ||"], r["||Δ||2"]), reverse=True)

        print("\n=== Summary ===")
        print(f"Compared tensors (float): {total_tensors}")
        print(f"Tensors changed (max|Δ|>0): {changed_tensors}")
        print(f"Elements total: {total_elems:,}")
        print(f"Elements |Δ|>{change_thresh}: {changed_elems:,} "
              f"({changed_elems / total_elems * 100:.4f}% if total>0)")

        print(f"\nTop {min(topk, len(rows_sorted))} tensors by relative change:")
        for r in rows_sorted[:topk]:
            print(f"- {r['name']:<80} {str(r['shape']):>16}  "
                  f"rel||Δ||={r['rel||Δ||']:.6g}  ||Δ||2={r['||Δ||2']:.6g}  "
                  f"max|Δ|={r['max|Δ|']:.6g}  changed={r['changed_frac']:.2%}")

    # --- call it on your diffusion model submodule ---
    report_weight_diffs(
        model_after=model.model.diffusion_model,  # after LoRA applied
        model_before=model_orig.model.diffusion_model,
        topk=30,
        eps=1e-12,
        change_thresh=0.0,  # set e.g. to 1e-12 if you want to ignore tiny fp noise
    )


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

                # Compute epsilon predictions for both models
                model.current_conditioning = cond
                eps_lora = model.apply_model(z_batch, t_enc_ddpm, cond)
                model.current_conditioning = cond_orig
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
