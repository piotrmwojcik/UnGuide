import os
import json
import argparse
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
import time
from torchvision.utils import save_image
from torchvision.transforms.functional import to_pil_image

from generate_images import decide_w, load_model_from_config, AutoGuidedModel
from ldm.models.diffusion.ddimcopy import DDIMSampler
from sampling import sample_model
from utils import apply_lora_to_model, set_seed

def compute_latent_diff(
    prompt,
    model,
    model_orig,
    sampler_orig,
    guidance,
    seed,
    start_codes,
    repeats=30,
    batch_size=1,
    image_size=512,
    t_enc=40,
    og_num=None,
    og_num_lim=None,
    ddim_steps=50,
    ddim_eta=0.0,
    device="cuda",
):
    if og_num is None:
        og_num = round((t_enc / ddim_steps) * 1000)
    if og_num_lim is None:
        og_num_lim = round(((t_enc + 1) / ddim_steps) * 1000)
    diffs_all = []

    set_seed(seed)
    gen = torch.Generator(device=device).manual_seed(seed)

    cond = model.get_learned_conditioning([prompt] * batch_size)
    cond_orig = model_orig.get_learned_conditioning([prompt] * batch_size)
    
        
    t_enc_ddpm = torch.randint(
        og_num, og_num_lim, (batch_size,), generator=gen, device=device
    )

    with torch.no_grad():
        z_batch = sample_model(
            model_orig,
            sampler_orig,
            cond_orig,
            image_size,
            image_size,
            ddim_steps,
            guidance,
            ddim_eta,
            start_code=start_codes,
            n_samples=batch_size,
            till_T=t_enc,
            verbose=False,
        )

        eps_lora = model.apply_model(z_batch, t_enc_ddpm, cond)
        eps_orig = model_orig.apply_model(z_batch, t_enc_ddpm, cond_orig)
        diffs = (
            (eps_lora - eps_orig)
            .view(batch_size, -1)
            .norm(dim=1)
            .cpu()
            .numpy()
            .tolist()
        )
        #diffs_all.extend(diffs)

    return diffs


def generate_image_cfg_auto(
    model, start_code, prompt, steps=50, guidance_scale=7.5, device="cuda"
):
    model = model.eval()
    cond = model.get_learned_conditioning([prompt])
    uncond = model.get_learned_conditioning([""])
    sampler = DDIMSampler(model=model)

    with torch.no_grad():
        with torch.autocast(device_type="cuda", dtype=torch.float16):
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

            decoded = model.decode_first_stage(samples)
            decoded = (decoded + 1.0) / 2.0
            decoded = torch.clamp(decoded, 0.0, 1.0)

    return decoded


def generate_with_dynamic_w(
    prompt,
    model,
    model_orig,
    shape=(4, 64, 64),
    steps=50,
    guidance_scale=7.5,
    w=0,
    gen=None,
    device="cuda",
):
    start_code = torch.randn(1, *shape, generator=gen, device=device)
    auto_model = AutoGuidedModel(
        model_full=model_orig, model_unlearned=model, w=w, cfg_scale=guidance_scale
    )
    img = generate_image_cfg_auto(
        auto_model, start_code, prompt, steps, guidance_scale, device
    )
    img_np = img[0].detach().cpu().permute(1, 2, 0).numpy()
    img_pil = to_pil_image((img_np * 255).astype(np.uint8))
    return img_pil


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
    parser.add_argument(
        "--w1", type=float, default=-1.0, help="W1"
    )
    parser.add_argument(
        "--w2", type=float, default=2.0, help="W2"
    )
    parser.add_argument("--ddim_eta", type=float, default=0.0)
    parser.add_argument("--batch_size", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()
    print("Start", flush=True)
    print(f"Using device: {args.device}")
    if os.path.join(args.output_dir, "train_config.json"):
        dirs = [args.output_dir]
    else:
        dirs = os.listdir(args.output_dir)

    for dirname in dirs:
        # Load prompts
        df = pd.read_csv(args.csv_path, index_col=0)
        exp_name = "_".join([str(item) for item in ["w1", args.w1, "w2", args.w2, "repeats", args.repeats, "ddim_steps", args.ddim_steps, "t_enc", args.t_enc]])
        exp_dirpath = os.path.join(args.output_dir, dirname)
        os.makedirs(os.path.join(exp_dirpath, "images", exp_name), exist_ok=True)
        lora_path = os.path.join(exp_dirpath, "models", "lora.pth")
        if not os.path.exists(lora_path):
            print(f"Skip {dirname} - lora.pth not found")
            continue
        if len(os.listdir(os.path.join(exp_dirpath, "images"))) >= len(df):
            print(f"Skip {dirname} - already processed")
            continue
        print(f"Processing experiment: {dirname}.", flush=True)
        print("images", len(os.listdir(os.path.join(exp_dirpath, "images", exp_name))), flush=True)
        

        # Load and prepare models
        model_orig = load_model_from_config(args.config, args.ckpt, args.device)
        model = load_model_from_config(args.config, args.ckpt, args.device)
        lora_state_dict = torch.load(lora_path, map_location="cuda")
        apply_lora_to_model(model.model.diffusion_model, lora_state_dict, alpha=args.alpha)
        sampler_orig = DDIMSampler(model_orig)

        # Precompute limits
        og_num = round((args.t_enc / args.ddim_steps) * 1000)
        og_num_lim = round(((args.t_enc + 1) / args.ddim_steps) * 1000)

        # Iterate over prompts
        
        for image_id, row in df.iterrows():
            image_path = os.path.join(exp_dirpath, "images", exp_name, f"{image_id:05d}.jpg")
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

            prompt_diffs_arr = []
            empty_diffs_arr = []
            for repeat in range(args.repeats):
                start_codes = torch.randn(
                    (args.batch_size, 4, args.image_size // 8, args.image_size // 8),
                    generator=gen,
                    device=args.device,
                )   
                prompt_diff = compute_latent_diff(
                    prompt,
                    model,
                    model_orig,
                    sampler_orig,
                    guidance,
                    seed,
                    start_codes=start_codes,
                    repeats=args.repeats,
                    batch_size=args.batch_size,
                    image_size=args.image_size,
                    t_enc=args.t_enc,
                    og_num=og_num,
                    og_num_lim=og_num_lim,
                    ddim_steps=args.ddim_steps,
                    ddim_eta=args.ddim_eta,
                    device=args.device,
                )
                empty_diff = compute_latent_diff(
                    "",
                    model,
                    model_orig,
                    sampler_orig,
                    guidance,
                    seed,
                    start_codes=start_codes,
                    repeats=args.repeats,
                    batch_size=args.batch_size,
                    image_size=args.image_size,
                    t_enc=args.t_enc,
                    og_num=og_num,
                    og_num_lim=og_num_lim,
                    ddim_steps=args.ddim_steps,
                    ddim_eta=args.ddim_eta,
                    device=args.device,
                )
                prompt_diffs_arr.append(prompt_diff)
                empty_diffs_arr.append(empty_diff)
            prompt_diff = np.mean(prompt_diffs_arr, axis=0) 
            empty_diff = np.mean(empty_diffs_arr, axis=0)
            w = decide_w(prompt_diff, empty_diff, w1=args.w1, w2=args.w2)
            set_seed(seed)
            gen = torch.Generator(device=args.device).manual_seed(seed)
            img = generate_with_dynamic_w(
                prompt=prompt,
                model=model,
                model_orig=model_orig,
                shape=(4, args.image_size // 8, args.image_size // 8),
                steps=args.ddim_steps,
                guidance_scale=guidance,
                w=w,
                gen=gen,
                device=args.device,
            )

            
            img.save(image_path)
            end = time.time()
            print(f"Prompt [{image_id}] processed in {end - start:.2f}) seconds. Saved to {image_path}", flush=True)
