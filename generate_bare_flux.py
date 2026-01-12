import os
import argparse
import torch
import pandas as pd
from tqdm import tqdm
import math
import time
import numpy as np
import re
from tools.prompt_process import encode_prompt
from tools.scheduler_process import FlowMatchEulerDiscreteScheduler
from tools.ir_concept import UniversalModelCaller, MoE
from utils.esd_utils import latent_sample, predict_noise, flux_pack_latents, _prepare_latent_image_ids
from diffusers.utils.torch_utils import randn_tensor
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

API_KEY=''
END_POINT='https://research-01-02.openai.azure.com/'
API_VERSION = "2024-08-01-preview"

api_keys = {
    "gpt": {"azure":True, "api_key":API_KEY, "end_point":END_POINT, "api_version": API_VERSION},
    "claude": None,
    "kimi": None,
    "qwen": None
}

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

@torch.no_grad()
def inference_latent_sample(transformer, scheduler, batch_size, num_channels_latents, height, width, prompt_embeds,
                  pooled_prompt_embeds, text_ids, guidance, timesteps, vae_scale_factor, latents=None):
    
    height = int(height) // 8
    width = int(width) // 8
    shape = (batch_size, num_channels_latents, height, width)

    if latents is None:
        latents = randn_tensor(shape, generator=None, dtype=torch.bfloat16, device=transformer.device)
    
    latents = flux_pack_latents(latents, batch_size, num_channels_latents, height, width)
    
    latent_image_ids = _prepare_latent_image_ids(batch_size, height // 2, width // 2, transformer.device,
                                                 torch.bfloat16)

    #print(timesteps)
    #scheduler.set_train_timesteps(timesteps, device=transformer.device, linear=True)
    #timesteps_tensor = scheduler.timesteps
    #print(timesteps_tensor)

    latents = latents.to(transformer.device).bfloat16()
    pooled_prompt_embeds = pooled_prompt_embeds.bfloat16()
    prompt_embeds = prompt_embeds.bfloat16()
    if text_ids.dim() == 3:
        text_ids = text_ids[0]
    elif text_ids.dim() == 2:
        text_ids = text_ids
    else:
        raise ValueError(f"Unexpected txt_ids shape: {text_ids.shape}")

    text_ids = text_ids.to(dtype=torch.bfloat16)

    sigmas = np.linspace(1.0, 1 / timesteps, timesteps)
    image_seq_len = latents.shape[1]
    mu = calculate_shift(
        image_seq_len,
        self.scheduler.config.base_image_seq_len,
        self.scheduler.config.max_image_seq_len,
        self.scheduler.config.base_shift,
        self.scheduler.config.max_shift,
    )
    timesteps_tensor, num_inference_steps = retrieve_timesteps(
        self.scheduler,
        timesteps,
        device,
        timesteps,
        sigmas,
        mu=mu,
    )

    for i, t in enumerate(timesteps_tensor):
        timestep = t.expand(latents.shape[0]).to(torch.bfloat16)

        noise_pred = transformer(
            hidden_states=latents,
            timestep=timestep / 1000,
            guidance=guidance,
            pooled_projections=pooled_prompt_embeds,
            encoder_hidden_states=prompt_embeds,
            txt_ids=text_ids,
            img_ids=latent_image_ids,
            return_dict=False,
        )

        if isinstance(noise_pred, (tuple, list)):
            noise_pred = noise_pred[0]
            
        latents = scheduler.step(noise_pred, t, latents, return_dict=False)[0]

    return latents, latent_image_ids

@torch.no_grad()
def generate_one_image_from_prompt(
    prompt: str,
    *,
    transformer,
    vae,
    noise_scheduler,
    text_encoders,
    tokenizers,
    height: int = 512,
    width: int = 512,
    num_inference_steps: int = 28,
    weight_dtype: torch.dtype = torch.bfloat16,
    seed: int | None = None,
):
    """
    Training-free single-image generation using your FLUX components.
    - Uses ONLY one text prompt (no negative prompt, no losses).
    - Samples latents with `latent_sample(...)` and decodes with VAE (shift/scaling aware).
    Returns: PIL.Image
    """

    device = transformer.device
    bsz = 1

    # Optional deterministic seed
    if seed is not None:
        torch.manual_seed(seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(seed)

    # --- Text embeddings (positive prompt only) ---
    prompts = [prompt]
    emb_p, pooled_emb_p, text_ids_p = compute_text_embeddings(
        prompts, text_encoders, tokenizers
    )

    # --- VAE scale factor (same as your snippet) ---
    vae_scale_factor = 2 ** (len(vae.config.block_out_channels))

    # --- Sample latents from pure noise using your sampler ---
    # NOTE: latent_sample signature from your snippet:
    # latent_sample(transformer, noise_scheduler, batch_size, num_channels, height, width,
    #               emb, pooled_emb, text_ids, guidance, steps, vae_scale_factor)
    num_channels = vae.config.latent_channels

    image_h = 512
    image_w = 512

    latent_h = image_h // vae_scale_factor
    latent_w = image_w // vae_scale_factor

    # --- Create dummy latent ---
    model_input = torch.zeros(
        (bsz, num_channels, latent_h, latent_w),
        device=vae.device,
        dtype=weight_dtype,
    )

    # --- Apply the same normalization as real latents ---
    model_input = (model_input - vae.config.shift_factor) * vae.config.scaling_factor

    start_guidance = 3
    start_guidance = torch.tensor([start_guidance], device=transformer.device)
    start_guidance = start_guidance.expand(model_input.shape[0])

    with torch.no_grad():
        z, latent_image_ids = inference_latent_sample(
            transformer,
            noise_scheduler,
            bsz,
            num_channels,
            height,
            width,
            emb_p.to(device),
            pooled_emb_p.to(device),
            text_ids_p.to(device),
            start_guidance,
            int(num_inference_steps),
            vae_scale_factor,
        )
    # If your latent_sample returns packed latents, unpack them.
    # (If it already returns (B,C,H,W), this branch will be skipped.)
    print('!!! ', z.shape)
    bsz, seq_len, ch = z.shape  # (1, 1024, 64)
    side = int(math.isqrt(seq_len))
    assert side * side == seq_len, f"seq_len={seq_len} not square"

    latent_h = side * 2  # 64
    latent_w = side * 2  # 64
    img_h = latent_h * vae_scale_factor  # 512 if scale=8
    img_w = latent_w * vae_scale_factor  # 512

    z = FluxPipeline._unpack_latents(
        z,
        height=img_h,
        width=img_w,
        vae_scale_factor=vae_scale_factor,
    )

    # --- Decode latents with VAE (invert shift/scaling) ---
    shift = vae.config.shift_factor
    scale = vae.config.scaling_factor

    z = z / scale + shift

    # decode in fp32 (matches VAE bias dtype)
    z = z.to(device=vae.device, dtype=torch.float32)
    vae = vae.to(device=vae.device, dtype=torch.float32)

    decoded = vae.decode(z).sample
    img = (decoded / 2 + 0.5).clamp(0, 1)

    # Convert to PIL
    import numpy as np
    from PIL import Image

    img = img[0].permute(1, 2, 0).float().cpu().numpy()
    img = (img * 255).round().astype(np.uint8)
    return Image.fromarray(img)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate images with base Flux from CSV")
    parser.add_argument("--csv_path", type=str, default="data/I2P_prompts_4703.csv")
    parser.add_argument("--pretrained_model_name_or_path", type=str, default="black-forest-labs/FLUX.1-dev")
    parser.add_argument("--output_dir", type=str, default="generated_bare_flux")
    parser.add_argument("--save_folder", type=str, default="images")
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--num_inference_steps", type=int, default=28)
    parser.add_argument("--nudity", type=bool, default=True)
    parser.add_argument("--max_sequence_length", type=int, default=256)
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

    noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
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

    weight_dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    transformer = FluxTransformer2DModel.from_pretrained(
        args.pretrained_model_name_or_path, torch_dtype=weight_dtype,
        subfolder="transformer", revision=None, variant=None
    ).to(device)

    transformer.requires_grad_(False)
    vae.requires_grad_(False)
    text_encoder_one.requires_grad_(False)
    text_encoder_two.requires_grad_(False)

    tokenizers = [tokenizer_one, tokenizer_two]
    text_encoders = [text_encoder_one, text_encoder_two]

    def compute_text_embeddings(prompts, text_encoders, tokenizers):
        # prompts: List[str] or str
        if isinstance(prompts, str):
            prompts = [prompts]

        prompt_embeds, pooled_prompt_embeds, text_ids = encode_prompt(
            text_encoders, tokenizers, prompts, args.max_sequence_length
        )
        return (
            prompt_embeds.to(transformer.device),
            pooled_prompt_embeds.to(transformer.device),
            text_ids.to(transformer.device),
        )

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

        image = generate_one_image_from_prompt(
            prompt=prompt,
            transformer=transformer,
            vae=vae,
            noise_scheduler=noise_scheduler,
            text_encoders=text_encoders,
            tokenizers=tokenizers,
            height=args.image_size,
            width=args.image_size,
            num_inference_steps=args.num_inference_steps,
            weight_dtype=weight_dtype,
            seed=seed,  # uses your per-row seed
        )

        image.save(image_path)

        images_generated += 1
        end = time.time()
        print(f"Prompt [{prompt}] processed in {end - start:.2f} seconds. Saved to {image_path}")