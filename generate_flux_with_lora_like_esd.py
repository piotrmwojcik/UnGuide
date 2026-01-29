import os
import argparse
import random
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import time
from functools import partial
import re
import dotenv
dotenv.load_dotenv()
from diffusers import FluxPipeline, AutoencoderKL, FlowMatchEulerDiscreteScheduler
from diffusers.models import FluxTransformer2DModel
from accelerate.utils import set_seed as hf_set_seed
from huggingface_hub import login
from transformers import CLIPTextModel, CLIPTokenizer, PretrainedConfig, T5TokenizerFast

from hyper_lora import HyperLoRALinear, HypernetworkManager, inject_hyper_lora

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

def import_model_class_from_model_name_or_path(pretrained_model_name_or_path: str, revision: str = None, subfolder: str = "text_encoder"):
        text_encoder_config = PretrainedConfig.from_pretrained(
            pretrained_model_name_or_path, subfolder=subfolder, revision=revision
        )
        model_class = text_encoder_config.architectures[0]
        if model_class == "CLIPTextModel":
            from transformers import CLIPTextModel
            return CLIPTextModel
        elif model_class == "T5EncoderModel":
            from transformers import T5EncoderModel
            return T5EncoderModel
        else:
            raise ValueError(f"{model_class} is not supported.")


def load_flux_models(basemodel_id="black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16, device='cuda:0'):
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(basemodel_id, subfolder="scheduler")

    text_encoder_cls_one = import_model_class_from_model_name_or_path(basemodel_id, None)
    text_encoder_cls_two = import_model_class_from_model_name_or_path(basemodel_id, None, subfolder="text_encoder_2")

    text_encoder = text_encoder_cls_one.from_pretrained(basemodel_id, subfolder="text_encoder", torch_dtype=torch_dtype)
    text_encoder_2 = text_encoder_cls_two.from_pretrained(basemodel_id, subfolder="text_encoder_2", torch_dtype=torch_dtype)

    tokenizer = CLIPTokenizer.from_pretrained(basemodel_id, subfolder="tokenizer")
    tokenizer_2 = T5TokenizerFast.from_pretrained(basemodel_id, subfolder="tokenizer_2")

    vae = AutoencoderKL.from_pretrained(basemodel_id, subfolder="vae", torch_dtype=torch_dtype)
    transformer = FluxTransformer2DModel.from_pretrained(basemodel_id, subfolder="transformer", torch_dtype=torch_dtype)

    pipe = FluxPipeline(
        transformer=transformer,
        scheduler=scheduler,
        vae=vae,
        text_encoder=text_encoder,
        text_encoder_2=text_encoder_2,
        tokenizer=tokenizer,
        tokenizer_2=tokenizer_2,
    )
    return pipe, transformer, scheduler


def load_lora_weights(model_wrapper, lora_path, device, check_keys=5):
    # Accept both transformer or pipe as input
    if hasattr(model_wrapper, 'transformer'):
        transformer = model_wrapper.transformer
    else:
        transformer = model_wrapper

    # real tensors
    tensor_map = {n: p for n, p in transformer.named_parameters()}
    tensor_map.update({n: b for n, b in transformer.named_buffers()})

    lora_state_dict = torch.load(lora_path, map_location="cpu")

    # Compatible with both accelerator.save and torch.save (plain dict)
    if isinstance(lora_state_dict, dict):
        # Accept plain dict (torch.save from train_flux_like_esd.py)
        if 'state_dict' in lora_state_dict:
            lora_state_dict = lora_state_dict['state_dict']
        elif 'module' in lora_state_dict:
            lora_state_dict = lora_state_dict['module']
        # If it's a flat dict of tensors, use as is
    if not isinstance(lora_state_dict, dict):
        raise ValueError(f"Loaded LoRA checkpoint is not a dict: {type(lora_state_dict)}")

    # pick a few keys that exist in both
    #print(tensor_map.keys())
    print(lora_state_dict.keys())
    #print(tensor_map.keys())
    common = [k for k in lora_state_dict.keys() if k in tensor_map]
    print("ckpt keys:", len(lora_state_dict), "common keys:", len(common))
    print("example common keys:", common[:10])

    with torch.no_grad():
        for i, k in enumerate(common):
            t = tensor_map[k]
            v = lora_state_dict[k].to(device=t.device, dtype=t.dtype)

            before = t.detach().float().norm().item()
            diff = (t.detach().float() - v.detach().float()).norm().item()

            print(f"[BEFORE] {k}: ||t||={before:.4e}, ||t-v||={diff:.4e}")

            t.copy_(v)

            after = t.detach().float().norm().item()
            diff2 = (t.detach().float() - v.detach().float()).norm().item()
            print(f"[AFTER ] {k}: ||t||={after:.4e}, ||t-v||={diff2:.4e}")

    return transformer



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate images with base Flux from CSV")
    parser.add_argument("--csv_path", type=str, default="data/I2P_prompts_4703.csv")
    parser.add_argument("--output_dir", type=str, default="generated_results_flux_lora")
    parser.add_argument("--save_folder", type=str, default="images")
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--num_inference_steps", type=int, default=28)
    parser.add_argument("--nudity", type=bool, default=True)
    parser.add_argument("--guidance_scale", type=float, default=3.0)
    parser.add_argument("--n_images", type=int, default=None)
    parser.add_argument("--lora_path", type=str, default=None)
    parser.add_argument("--rank", type=int, default=9,
                       help="LoRA rank (must match training config)")
    parser.add_argument("--lora_alpha", type=float, default=9.0,
                       help="LoRA alpha (must match training config)")
    parser.add_argument("--hyper_train_steps", type=int, default=300,
                       help="Hypernetwork timesteps (must match training config)")
    parser.add_argument("--use_pooler", type=bool, default=True,
                       help="Use CLIP pooler output")
    parser.add_argument("--use_orig_concat", type=bool, default=False,
                       help="Use original concat in HyperLoRA (must match training config)")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=2024)
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    torch.set_num_threads(torch.get_num_threads())

    # Load Flux pipeline

    # Load models exactly as in train_flux_like_esd.py
    pipe, transformer, scheduler = load_flux_models(
        basemodel_id="black-forest-labs/FLUX.1-dev",
        torch_dtype=torch.bfloat16,
        device=device
    )
    pipe = pipe.to(device)
    pipe_device = device

    # Load prompts
    df = pd.read_csv(args.csv_path, index_col=0)

    save_dir = os.path.join(args.output_dir, args.save_folder)
    os.makedirs(save_dir, exist_ok=True)

    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
    clip_text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14").to(pipe_device).eval()

    print("Setting up HyperLoRA...")
    transformer.hyper = HypernetworkManager()

    clip_size = 768 if args.use_pooler else 512
    target_modules = ["attn.add_v_proj", "attn.to_v", "attn.to_out.0"]

    hyper_lora_factory = partial(
        HyperLoRALinear,
        clip_size=clip_size,
        rank=args.rank,
        alpha=args.lora_alpha,
        train_steps=args.hyper_train_steps,
        use_orig_concat=args.use_orig_concat
    )

    hyper_lora_layers = inject_hyper_lora(
        transformer, target_modules, hyper_lora_factory
    )

    for layer_name, layer in hyper_lora_layers:
        layer.set_parent_model(transformer)
        transformer.hyper.add_hyperlora(layer_name, layer.hyper_lora)

    print(f"Injected HyperLoRA into {len(hyper_lora_layers)} layers")

    load_lora_weights(transformer, args.lora_path, device)

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
        image_path = os.path.join(save_dir, f"{image_id:05d}.png")
        if os.path.exists(image_path):
            continue
        prompt = coerce_prompt(row.get("prompt", ""))
        if not isinstance(prompt, str) or not prompt.strip():
            print(f"Skip [{image_id}] empty prompt")
            continue

        weight_dtype = torch.bfloat16

        # Get the device where HyperLoRA layers are located
        hyper_device = (
            transformer.hyper.hyper_layers[0].alpha.device
            if transformer.hyper.hyper_layers
            else torch.device("cpu")
        )

        with torch.no_grad():
            inputs = tokenizer(
                prompt,
                max_length=tokenizer.model_max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            ).to(pipe_device).input_ids

            with torch.no_grad():
                if args.use_pooler:
                    context_emb = clip_text_encoder(inputs).pooler_output.detach()
                else:
                    context_emb = clip_text_encoder(inputs).last_hidden_state.detach()


        context_emb = context_emb.to(device=hyper_device)
        timestep = torch.tensor([args.hyper_train_steps], device=hyper_device)

        STEP = 300
        hyper_device = transformer.hyper.hyper_layers[0].alpha.device if transformer.hyper.hyper_layers else "cpu"
        transformer.hyper.set_context(context_emb.to(device=hyper_device),
                         torch.tensor([STEP], device=hyper_device))
        transformer.hyper.compute_and_cache_loras(context_emb.to(device=hyper_device),
                           torch.tensor([STEP], device=hyper_device))

        print("cache size:", len(transformer.hyper.lora_weights_cache))
        print("example cache key:", next(iter(transformer.hyper.lora_weights_cache.keys())))

        seed = int(row.get("evaluation_seed", 0))
        generator = torch.Generator(device).manual_seed(seed)

        start = time.time()
        image = pipe(
            prompt=prompt,
            guidance_scale=3,
            num_inference_steps=args.num_inference_steps,
            height=args.image_size,
            width=args.image_size,
            generator=generator,
            max_sequence_length=256
        ).images[0]
        image.save(image_path)
        images_generated += 1
        end = time.time()
        print(f"Prompt [{prompt}] processed in {end - start:.2f} seconds. Saved to {image_path}")
