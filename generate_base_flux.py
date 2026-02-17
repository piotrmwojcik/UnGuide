import os
import argparse
import torch
import pandas as pd
from tqdm import tqdm
import time
import re
from diffusers import FluxPipeline
from huggingface_hub import login

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
token = os.environ.get("HF_TOKEN")
if token:
    login(token)
else:
    print("Warning: HF_TOKEN not set.")

def coerce_prompt(v):
    # Treat None/NaN as empty
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return ""

    # Real list/tuple/set -> comma-separated string
    if isinstance(v, (list, tuple, set)):
        return ", ".join(str(x).strip() for x in v if str(x).strip())

    # String cases
    s = str(v).strip()

    # Drop an optional "Prompt" label like "Prompt [a, b]" or "Prompt: a, b"
    s = re.sub(r'^\s*prompt\s*[:\-]?\s*', "", s, flags=re.I)

    # If it's bracketed like "[a, b]" without quotes, normalize it
    m = re.match(r'^\[\s*(.*)\s*\]$', s)
    if m:
        parts = [p.strip() for p in m.group(1).split(",")]
        return ", ".join(p for p in parts if p)

    return s


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate images with base Flux from CSV")
    parser.add_argument("--csv_path", type=str, default="data/I2P_prompts_4703.csv")
    parser.add_argument("--output_dir", type=str, default="generated_base_flux_i2p")
    parser.add_argument("--save_folder", type=str, default="images")
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--num_inference_steps", type=int, default=28)
    parser.add_argument("--nudity", type=bool, default=True)
    parser.add_argument("--guidance_scale", type=float, default=3.5)
    parser.add_argument("--n_images", type=int, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Load Flux pipeline
    cache_dir = "./models"
    os.makedirs(cache_dir, exist_ok=True)
    pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16, cache_dir=cache_dir)
    #pipe.enable_model_cpu_offload()
    pipe = pipe.to(device)

    # Load prompts
    df = pd.read_csv(args.csv_path, index_col=0)

    if args.nudity and "nudity_percentage" in df.columns:
        # Ensure numeric values
        df["nudity_percentage"] = pd.to_numeric(
            df["nudity_percentage"], errors="coerce"
        )

        # Keep only rows with non-zero nudity
        df = df[df["nudity_percentage"] > 0]

        # Sort by highest nudity first
        df = df.sort_values(by="nudity_percentage", ascending=False)

    save_dir = os.path.join(args.output_dir, args.save_folder)
    os.makedirs(save_dir, exist_ok=True)

    images_generated = 0
    for image_id, row in tqdm(df.iterrows(), total=len(df)):
        if args.n_images is not None and images_generated >= args.n_images:
            break
        image_path = os.path.join(save_dir, f"{int(image_id):05d}.png")
        if os.path.exists(image_path):
            continue

        prompt = coerce_prompt(row.get("prompt", ""))
        if not isinstance(prompt, str) or not prompt.strip():
            print(f"Skip [{int(image_id)}] empty prompt")
            continue

        #seed = 2024#int(row.get("evaluation_seed", 0))
        generator = torch.Generator(device).manual_seed(seed)

        start = time.time()
        image = pipe(
            prompt=prompt,
            guidance_scale=3.5,
            num_inference_steps=args.num_inference_steps,
            height=args.image_size,
            width=args.image_size,
            generator=generator,
            max_sequence_length=256
        ).images[0]
        image.save(image_path)
        images_generated += 1
        end = time.time()
        print(f"Prompt [{prompt}] processed in {end - start:.2f} seconds. Seed: {seed}. Saved to {image_path}")