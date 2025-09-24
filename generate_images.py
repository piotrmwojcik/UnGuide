import os
import json
import argparse
import torch
from ldm.models.diffusion.ddimcopy import DDIMSampler
from utils import load_model_from_config, apply_lora_to_model, set_seed
from torchvision.transforms.functional import to_pil_image
from autoguide import AutoGuidedModel
import numpy as np
import time
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(
        description="Auto-guided image generation with Stable Diffusion and LoRA"    )
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


if __name__ == "__main__":
    RANK = int(os.environ.get("RANK", "0"))
    WORLD_SIZE = int(os.environ.get("WORLD_SIZE", "1"))

    args = parse_args()

    exps = os.listdir(args.output_dir)
    print(f"Exps: {exps}", flush=True)
    for exp in exps:
        exp_filepath = os.path.join(args.output_dir, exp)
        img_root = os.path.join(args.output_dir, exp, "images")
        lora_filepath = os.path.join(exp_filepath, "models", "hyper_lora.pth")
       

        diff_results_path = os.path.join(exp_filepath, "calc_diff_results.json")
        train_json_path = os.path.join(exp_filepath, "train_config.json")
        
        with open(diff_results_path, 'r') as f:
            results = json.load(f)

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
                if f.lower().endswith(('.png','.jpg','.jpeg','.bmp','.tiff'))
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
        lora_state_dict = torch.load(lora_filepath, map_location=args.device)
        apply_lora_to_model(model_unl.model.diffusion_model, lora_state_dict, alpha=8)

        for prompt in prompts:
            class_name = prompt.split(" ")[-1]
            class_root = os.path.join(img_root, class_name)
            os.makedirs(class_root, exist_ok=True)
            if len(os.listdir(class_root)) == args.samples:
                continue

            w = decide_w(
                results["prompt_avgs"].get(prompt), results["prompt_avgs"].get(""),
                w1=args.w1, w2=args.w2
            )
            # Prepare models and sampler
            auto_model = AutoGuidedModel(
                model_full, model_unl, w=w
            ).eval()
            sampler = DDIMSampler(model=auto_model)

            # Conditioning
            cond = auto_model.get_learned_conditioning([prompt])
            uncond = auto_model.get_learned_conditioning([""])
                
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
                img = generate_image(
                    sampler, auto_model, start_code, cond, uncond, args.steps
                )
                img_np = img[0].cpu().permute(1, 2, 0).numpy()
                img_pil = to_pil_image((img_np * 255).astype(np.uint8))

                
                img_pil.save(filename_path, format='JPEG', quality=90, optimize=True)
                end = time.time()
                print(f"Generate: {end - start}", flush=True)
