import os
import argparse
import torch
import pandas as pd
from tqdm import tqdm
import time
import re
from tools.prompt_process import encode_prompt
from tools.scheduler_process import CustomFlowMatchEulerDiscreteScheduler
from tools.ir_concept import UniversalModelCaller, MoE
import copy
from diffusers import FluxPipeline
from diffusers import (
    AutoencoderKL,
    FlowMatchEulerDiscreteScheduler,
    FluxPipeline,
    FluxTransformer2DModel,
)
from transformers import CLIPTokenizer, PretrainedConfig, T5TokenizerFast
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


def load_text_encoders(class_one, class_two, args):
    text_encoder_one = class_one.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder", revision=None, variant=None
    )
    text_encoder_two = class_two.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder_2", revision=None, variant=None
    )
    return text_encoder_one, text_encoder_two


def import_model_class_from_model_name_or_path(
    pretrained_model_name_or_path: str, revision: str, subfolder: str = "text_encoder"
):
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate images with base Flux from CSV")
    parser.add_argument("--csv_path", type=str, default="data/I2P_prompts_4703.csv")
    parser.add_argument("--pretrained_model_name_or_path", type=str, default="black-forest-labs/FLUX.1-dev")
    parser.add_argument("--output_dir", type=str, default="generated_base_flux")
    parser.add_argument("--save_folder", type=str, default="images")
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--nudity", type=bool, default=True)
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--n_images", type=int, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    tokenizer_one = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer",
        revision=None,
    )
    tokenizer_two = T5TokenizerFast.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer_2",
        revision=None,
    )

    # import correct text encoder classes
    text_encoder_cls_one = import_model_class_from_model_name_or_path(
        args.pretrained_model_name_or_path, None
    )
    text_encoder_cls_two = import_model_class_from_model_name_or_path(
        args.pretrained_model_name_or_path, None, subfolder="text_encoder_2"
    )

    noise_scheduler = CustomFlowMatchEulerDiscreteScheduler.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="scheduler"
    )
    noise_scheduler_copy = copy.deepcopy(noise_scheduler)
    text_encoder_one, text_encoder_two = load_text_encoders(text_encoder_cls_one, text_encoder_cls_two, args)
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="vae",
        revision=None,
        variant=None,
    )

    transformer = FluxTransformer2DModel.from_pretrained(
        args.pretrained_model_name_or_path, torch_dtype=torch.bfloat16,
        subfolder="transformer", revision=None, variant=None
    ).to(device)

    transformer.requires_grad_(False)
    vae.requires_grad_(False)
    text_encoder_one.requires_grad_(False)
    text_encoder_two.requires_grad_(False)

    # Load Flux pipeline
    #cache_dir = "./models"
    #os.makedirs(cache_dir, exist_ok=True)
    #pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16, cache_dir=cache_dir)
    #pipe = pipe.to(device)

    # Load prompts
    df = pd.read_csv(args.csv_path, index_col=0)

    # Check if this is an NSFW dataset with nudity_percentage column
    if args.nudity and "nudity_percentage" in df.columns:
        df["nudity_percentage"] = pd.to_numeric(df["nudity_percentage"], errors="coerce")
        df = df[df["nudity_percentage"].gt(0)]
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

        seed = int(row.get("evaluation_seed", 0))
        generator = torch.Generator(device).manual_seed(seed)

        start = time.time()
        # image = pipe(
        #     prompt=prompt,
        #     guidance_scale=args.guidance_scale,
        #     num_inference_steps=args.num_inference_steps,
        #     height=args.image_size,
        #     width=args.image_size,
        #     generator=generator,
        #     max_sequence_length=256
        # ).images[0]
        # image.save(image_path)
        images_generated += 1
        end = time.time()
        print(f"Prompt [{prompt}] processed in {end - start:.2f} seconds. Saved to {image_path}")