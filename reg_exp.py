import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from omegaconf import OmegaConf
from typing import Tuple
from torchvision.transforms.functional import to_pil_image
from PIL import ImageDraw, ImageFont
from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from utils import get_models
from transformers import CLIPModel, CLIPProcessor
from accelerate import Accelerator
from accelerate.utils import set_seed as hf_set_seed


# Embedding sampling functions

def _cap_cos_threshold_from_euclid(d: float) -> float:
    # On the unit sphere: ||x-y|| = d  <=>  cos >= 1 - d^2/2
    return 1.0 - 0.5 * (d ** 2)

@torch.no_grad()
def sample_seq_global_cap(base_seq: torch.Tensor, n: int, *,
                          cosine_threshold: float = None,
                          euclid_radius: float = None) -> torch.Tensor:
    """
    base_seq: [1,77,768] (float)  -- typically CLIPTextModel output after final LN
    Returns:  [n,77,768] sampled so that the FLATTENED vector stays within the cap.
    """
    assert base_seq.ndim == 3 and base_seq.shape[0] == 1
    B, T, D = base_seq.shape
    x = base_seq.reshape(1, T*D)
    if cosine_threshold is None:
        assert euclid_radius is not None
        tau = _cap_cos_threshold_from_euclid(euclid_radius)
    else:
        tau = float(cosine_threshold)
    
    x_norm = x.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    x_hat  = x / x_norm
    TD = x_hat.shape[-1]

    c = torch.rand(n, device=x.device, dtype=x.dtype) * (1 - tau) + tau
    xh = x_hat.expand(n, -1)
    g = torch.randn(n, TD, device=x.device, dtype=x.dtype)
    v = g - (g * xh).sum(dim=-1, keepdim=True) * xh
    v_hat = F.normalize(v, dim=-1)
    s = torch.sqrt(torch.clamp(1 - c**2, min=0))
    y_hat = c.unsqueeze(-1) * xh + s.unsqueeze(-1) * v_hat
    y = F.normalize(y_hat, dim=-1) * x_norm
    return y.view(n, T, D)

# Image generation and evaluation

@torch.no_grad()
def generate_and_save_sd_images_from_embed(
    model,
    sampler: DDIMSampler,
    clip_embedding: torch.Tensor,
    device: torch.device,
    steps: int = 50,
    eta: float = 0.0,
    out_dir: str = "tmp",
    prefix: str = "img_",
    start_code: torch.Tensor = None,  # [B,4,64,64]
    accelerator: "Accelerator" = None,
    save: bool = True,
):
    """
    Generates images with CFG from a CompVis SD model + DDIMSampler *using pre-computed embeddings*.
    """
    batch_size = clip_embedding.shape[0]
    
    if start_code is None:
        # Default to 512x512 noise shape
        start_code = torch.randn(batch_size, 4, 64, 64, device=device)
    else:
        assert start_code.shape[0] == batch_size, \
            f"start_code batch size ({start_code.shape[0]}) must match clip_embedding batch size ({batch_size})"
        start_code = start_code.to(device)

    model.eval()
    with torch.no_grad(), torch.autocast(device_type=device.type, enabled=(device.type == "cuda")):

        cond = clip_embedding.to(device)
        uncond = model.get_learned_conditioning([""] * batch_size)

        samples_latent, _ = sampler.sample(
            S=steps,
            conditioning={"c_crossattn": [cond]},
            batch_size=start_code.shape[0],
            shape=start_code.shape[1:],  # (4, 64, 64)
            verbose=False,
            unconditional_guidance_scale=7.5,
            unconditional_conditioning={"c_crossattn": [uncond]},
            eta=eta,
            x_T=start_code,
        )

        imgs = model.decode_first_stage(samples_latent)
        imgs = (imgs.clamp(-1, 1) + 1) / 2.0  # [B, C, H, W]

        # Gather across processes if accelerate is present
        if accelerator is not None:
            imgs_all = accelerator.gather(imgs)
        else:
            imgs_all = imgs

        do_save = save and (accelerator is None or accelerator.is_main_process)
        if do_save:
            out_path = Path(out_dir)
            out_path.mkdir(exist_ok=True, parents=True)
            for i, im in enumerate(imgs_all.cpu()):
                im_u8 = (im.clamp(0, 1) * 255).round().to(torch.uint8)
                pil_img = to_pil_image(im_u8)
                pil_img.save(out_path / f"{prefix}{i:04d}.png")

        # Return the tensor of images in [0,1], like generate_and_save_sd_images in train.py
        return imgs_all

def load_images_from_dir(dir_path: Path) -> list[Image.Image]:
    """Loads all .png images from a directory into a list of PIL Images."""
    image_files = sorted(list(dir_path.glob("*.png")))
    return [Image.open(f) for f in image_files]

@torch.no_grad()
def get_image_embeddings(
    images: list[Image.Image], 
    clip_model: CLIPModel, 
    clip_processor: CLIPProcessor,
    device: torch.device
) -> torch.Tensor:
    """Gets normalized CLIP image features for a list of PIL images."""
    inputs = clip_processor(images=images, return_tensors="pt").to(device)
    img_embeds = clip_model.get_image_features(**inputs)
    img_embeds = F.normalize(img_embeds, p=2, dim=-1) # Normalize for cosine similarity
    return img_embeds


def _save_collage_with_metrics(
    pil_images: list[Image.Image],
    target_prompt: str,
    p2p_sims: list[float],
    i2i_sims: list[float],
    out_path: Path,
    cols: int = 5,
    thumb_size: Tuple[int, int] = (256, 256),
):
    """Save a collage with the prompt at the top and per-image metrics under each thumbnail.

    pil_images: list of PIL Images (len = N)
    p2p_sims, i2i_sims: lists of floats length N
    """
    if len(pil_images) == 0:
        return

    N = len(pil_images)
    rows = (N + cols - 1) // cols
    pad = 8
    thumb_w, thumb_h = thumb_size

    # Estimate text height
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    title_h = 24
    metric_h = 36

    canvas_w = cols * thumb_w + (cols + 1) * pad
    canvas_h = title_h + pad + rows * (thumb_h + metric_h + pad) + pad

    canvas = Image.new("RGB", (canvas_w, canvas_h), color=(255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    # Draw title (centered)
    title_text = target_prompt
    draw.text((pad, pad), title_text, fill=(0, 0, 0), font=font)

    # Paste thumbnails and metrics
    for idx, im in enumerate(pil_images):
        r = idx // cols
        c = idx % cols
        x = pad + c * (thumb_w + pad)
        y = title_h + pad + r * (thumb_h + metric_h + pad)

        thumb = im.copy()
        thumb = thumb.convert("RGB")
        thumb.thumbnail((thumb_w, thumb_h), Image.LANCZOS)
        # center thumb in slot
        tx = x + max(0, (thumb_w - thumb.width) // 2)
        ty = y
        canvas.paste(thumb, (tx, ty))

        # metrics text
        p2p = p2p_sims[idx]
        i2i = i2i_sims[idx]
        txt = f"P2P: {p2p:.3f}  I2I: {i2i:.3f}"
        tx2 = x
        ty2 = y + thumb_h + 4
        draw.text((tx2, ty2), txt, fill=(0, 0, 0), font=font)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)

# Main experiment script

if __name__ == "__main__":
    
    # Configuration
    LDM_CONFIG_PATH = "./configs/stable-diffusion/v1-inference.yaml"
    LDM_CKPT_PATH = "./models/sd-v1-4.ckpt"
    CLIP_MODEL_ID = "openai/clip-vit-large-patch14"
    
    TARGET_PROMPT = "a photo of an airplane"
    
    N_SAMPLES = 10               # How many images to generate for each group
    COSINE_THRESH = '0.5,0.75'   
    SEED = 42                   
    DDIM_STEPS = 50
    
    # Output directories
    BASE_IMG_DIR = Path("output/base_images")
    REG_IMG_DIR = Path("output/reg_images")

    # Setup
    accelerator = Accelerator()
    device = accelerator.device
    accelerator.print(f"Using device: {device}")

    if not Path(LDM_CKPT_PATH).exists() or not Path(LDM_CONFIG_PATH).exists():
        accelerator.print("="*50)
        accelerator.print("ERROR: LDM_CONFIG_PATH or LDM_CKPT_PATH not found.")
        accelerator.print(f"Please update the paths at the top of the '{__file__}' script.")
        accelerator.print("="*50)
        exit(1)

    if SEED is not None:
        hf_set_seed(SEED)

    accelerator.print("Loading LDM/SD model (this may take a moment)...")
    model_orig, sampler_orig, _, _ = get_models(
        LDM_CONFIG_PATH, LDM_CKPT_PATH, accelerator.device
    )

    accelerator.print(f"Loading CLIP model '{CLIP_MODEL_ID}' for evaluation...")
    eval_clip_model = CLIPModel.from_pretrained(CLIP_MODEL_ID).to(accelerator.device)
    eval_clip_processor = CLIPProcessor.from_pretrained(CLIP_MODEL_ID)

    # Get Base and Regularization Embeddings
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cosine-thresholds",
        type=str,
        default=str(COSINE_THRESH),
        help="Comma-separated list of cosine thresholds to run (e.g. '0.9,0.95,0.98')",
    )
    args = parser.parse_args()

    print(f"Getting base embedding for: '{TARGET_PROMPT}'")
    # [1, 77, 768]
    base_seq = model_orig.get_learned_conditioning([TARGET_PROMPT])

    # Create a batch of the original embedding for comparison (used for base images)
    base_seq_batch = base_seq.expand(N_SAMPLES, -1, -1)
    start_code = torch.randn(N_SAMPLES, 4, 64, 64, device=device)

    # Parse thresholds list
    thresholds = [float(x) for x in args.cosine_thresholds.split(",")]

    # Generate Images
    print(f"\nGenerating {N_SAMPLES} base images (from original prompt)...")

    world_size = accelerator.num_processes if accelerator is not None else 1
    rank = accelerator.process_index if accelerator is not None else 0
    indices = list(range(rank, N_SAMPLES, world_size))
    local_n = len(indices)

    if local_n == 0:
        local_base_batch = torch.empty((0, base_seq.shape[1], base_seq.shape[2]), device=base_seq.device)
        local_start_code = torch.empty((0, 4, 64, 64), device=device)
    else:
        local_base_batch = base_seq.expand(N_SAMPLES, -1, -1)[indices]
        local_start_code = start_code[indices]

    # Each rank generates only its shard; generate_and_save will gather across ranks
    base_images = generate_and_save_sd_images_from_embed(
        model=model_orig,
        sampler=sampler_orig,
        clip_embedding=local_base_batch,
        device=device,
        accelerator=accelerator,
        steps=DDIM_STEPS,
        out_dir=BASE_IMG_DIR,
        prefix="base_",
        start_code=local_start_code,
    )

    # convert returned tensor to PIL images for CLIP processing (only main process has images)
    if isinstance(base_images, torch.Tensor):
        pil_base_images = [to_pil_image((im.clamp(0, 1) * 255).round().to(torch.uint8)) for im in base_images.cpu()]
    else:
        pil_base_images = base_images
    
    print(f"Generating {N_SAMPLES} regularization images (from sampled embeds)...")
    # Loop over thresholds and perform generation+metrics per threshold
    for tau in thresholds:
        print(f"\nSampling {N_SAMPLES} regularization embeddings with cos_thresh={tau}...")
        # Each rank samples only its local shard to avoid duplicate generation across ranks
        reg_embeddings_local = sample_seq_global_cap(base_seq, n=local_n, cosine_threshold=tau)
        if accelerator is not None:
            reg_embeddings_all = accelerator.gather(reg_embeddings_local)
        else:
            reg_embeddings_all = reg_embeddings_local

        # per-threshold output folder
        per_dir = Path(REG_IMG_DIR) / f"cos_{tau:.3f}"

        reg_images = generate_and_save_sd_images_from_embed(
            model=model_orig,
            sampler=sampler_orig,
            clip_embedding=reg_embeddings_local,
            device=device,
            accelerator=accelerator,
            steps=DDIM_STEPS,
            out_dir=per_dir,
            prefix="reg_",
            start_code=local_start_code,  # Use the same noise shard
        )

        if isinstance(reg_images, torch.Tensor):
            pil_reg_images = [to_pil_image((im.clamp(0, 1) * 255).round().to(torch.uint8)) for im in reg_images.cpu()]
        else:
            pil_reg_images = reg_images

        # ensure all processes have finished gathering/saving for this threshold
        if accelerator is not None:
            accelerator.wait_for_everyone()

        # Only main process computes embeddings/metrics and saves collage
        if accelerator is None or accelerator.is_main_process:
            print("\nCalculating image embeddings for similarity...")
            base_img_embeds = get_image_embeddings(pil_base_images, eval_clip_model, eval_clip_processor, device)
            reg_img_embeds = get_image_embeddings(pil_reg_images, eval_clip_model, eval_clip_processor, device)

            # Image-to-image (paired) similarities
            paired_i2i = (base_img_embeds * reg_img_embeds).sum(dim=-1).cpu().tolist()

            # Prompt-to-Prompt similarity: compute cosine between flattened base_seq and each reg_embedding
            base_flat = base_seq.reshape(-1)
            reg_flat = reg_embeddings_all.reshape(reg_embeddings_all.shape[0], -1)
            base_flat_n = F.normalize(base_flat, dim=0)
            reg_flat_n = F.normalize(reg_flat, dim=1)
            p2p = (reg_flat_n @ base_flat_n).cpu().tolist()

            # Print per-threshold summary
            avg_i2i = float(sum(paired_i2i) / len(paired_i2i)) if paired_i2i else float('nan')
            avg_p2p = float(sum(p2p) / len(p2p)) if p2p else float('nan')
            print(f"Threshold {tau:.3f} | Avg I2I: {avg_i2i:.4f} | Avg P2P: {avg_p2p:.4f}")

            # Save a collage of regularization images with metrics
            collage_path = per_dir / "reg_collage.png"
            _save_collage_with_metrics(pil_reg_images, TARGET_PROMPT, p2p, paired_i2i, collage_path)

    print("\nExperiment complete.")