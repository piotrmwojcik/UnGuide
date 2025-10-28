#!/usr/bin/env python3
# Auto-generated from Jupyter notebook: minimal_notebook (1).ipynb
# This script is a flattened concatenation of code cells with IPython magics removed.

#A notebook for testing stable diffusion

# --- cell separator ---


# --- cell separator ---

import argparse
import json
import os
import wandb
from functools import partial

import numpy as np
import torch
import matplotlib.pyplot as plt

from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from data_utils import  TargetReferenceDataset, collate_prompts
from torchvision.transforms.functional import to_pil_image
import numpy as np
from transformers import CLIPTextModel, CLIPTokenizer
import torch
from tqdm import tqdm
from ldm.models.diffusion.ddimcopy import DDIMSampler
from utils import get_models, print_trainable_parameters, set_seed

# --- cell separator ---

from pathlib import Path
PROJECT_ROOT = Path.cwd().parent
import sys, os
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
    print('Top of sys.path:', sys.path[0])

model_orig, sampler_orig, model, sampler = get_models(
    "../configs/stable-diffusion/v1-inference.yaml", "../models/sd-v1-4.ckpt", "cuda"
)

# --- cell separator ---

def generate_and_save_sd_images(
    model,
    sampler,
    cond,
    device: torch.device,
    steps: int = 50,
    eta: float = 0.0,
    batch_size: int = 1,
    out_dir: str = "tmp",
    prefix: str = "unl_",
    start_code: torch.Tensor = None,   # optional noise tensor [B,4,64,64] (for 512x512)
):
    """
    Generates images with CFG from a CompVis SD model + DDIMSampler and saves them.

    - model: Stable Diffusion model (CompVis LDM style)
    - sampler: DDIMSampler(model)
    - prompt: text prompt
    - device: torch.device("cuda") or torch.device("cpu")
    - steps: DDIM steps
    - eta: DDIM eta (0.0 => deterministic)
    - batch_size: number of samples to generate
    - out_dir: folder to save into
    - prefix: file prefix, e.g., 'unl_'
    - start_code: optional start noise of shape [B, 4, H/8, W/8]; if None, sampled internally.
                  For 512×512 set shape to [B, 4, 64, 64].
    """
    # derive latent shape from start_code or default to 512×512
    if start_code is None:
        start_code = torch.randn(batch_size, 4, 64, 64, device=device)  # 512x512

    # freeze & eval for safety

    with torch.no_grad(), torch.autocast(device_type=device.type, enabled=(device.type == "cuda")):
        uncond = model.get_learned_conditioning([""] * start_code.shape[0])

        samples_latent, _ = sampler.sample(
            S=steps,
            conditioning={"c_crossattn": [cond]},
            batch_size=start_code.shape[0],
            shape=start_code.shape[1:],  # (4, H/8, W/8)
            verbose=False,
            unconditional_guidance_scale=7.5,                 # CFG scale; tweak if needed
            unconditional_conditioning={"c_crossattn": [uncond]},
            eta=eta,
            x_T=start_code,
        )

        # decode latents to [0,1] images
        imgs = model.decode_first_stage(samples_latent)       # [-1, 1]
        imgs = (imgs.clamp(-1, 1) + 1) / 2.0                 # [0, 1]

        # save
        out_path = Path(out_dir)
        out_path.mkdir(exist_ok=True)
        for i, im in enumerate(imgs.cpu()):
            im_u8 = (im.clamp(0, 1) * 255).round().to(torch.uint8)  # [3,H,W]
            to_pil_image(im_u8).save(out_path / f"{prefix}{i:04d}.png")

        print(f"Saved {len(imgs)} image(s) to {out_path}/ with prefix '{prefix}'")
        return imgs  # [B,3,H,W] in [0,1]

# --- cell separator ---

sampler = DDIMSampler(model)

# --- cell separator ---

import torch
import torch.nn.functional as F

@torch.no_grad()
def sample_token_nearby(
    x: torch.Tensor,
    cosine_distance_max: float,   # e.g. 0.2  -> cos >= 0.8
    n: int = 1,                   # number of samples you want
    mode: str = "cap",            # "cap" (≤ dist_max) or "shell" (≈ equal distance)
):
    """
    x: [D] or [1,D] original token embedding (will be unit-normalized)
    cosine_distance_max: max allowed cosine distance (in [0, 2]); smaller = closer
    n: number of samples to draw
    mode:
      - "cap"   : cos ~ Uniform[tau, 1]            (anything within the cap)
      - "shell" : cos = tau (on the boundary)
    returns:
      y: [n, D] unit vectors with cosine(x, y) >= tau  (for "cap")
                 or cosine(x, y) == tau (for "shell"), up to float error
    """
    if x.ndim == 1:
        x = x.unsqueeze(0)
    device, dtype = x.device, x.dtype
    x_hat = F.normalize(x, dim=-1)              # [1,D]
    D = x_hat.shape[-1]

    # distance -> cosine threshold
    tau = 1.0 - float(cosine_distance_max)
    tau = max(-1.0, min(1.0, tau))              # clamp

    # expand base
    x_hat = x_hat.expand(n, -1)                  # [n,D]

    # pick cosines
    if mode == "cap":
        c = torch.rand(n, device=device, dtype=dtype) * (1 - tau) + tau
    elif mode == "shell":
        c = torch.full((n,), tau, device=device, dtype=dtype)
    else:
        raise ValueError("mode must be 'cap' or 'shell'")

    # random directions orthogonal to x
    g = torch.randn(n, D, device=device, dtype=dtype)
    proj = (g * x_hat).sum(dim=-1, keepdim=True) * x_hat
    v = g - proj
    v_hat = F.normalize(v, dim=-1)               # [n,D]

    # combine to hit desired cosine
    s = torch.sqrt(torch.clamp(1 - c**2, min=0))
    y = c.unsqueeze(-1) * x_hat + s.unsqueeze(-1) * v_hat
    return F.normalize(y, dim=-1)

# --- cell separator ---

tok = torch.randn(768)                   # example token
Y   = sample_token_nearby(tok, cosine_distance_max=0.2, n=4, mode="cap")
cos = (F.normalize(tok, dim=0).unsqueeze(0) @ Y.T).squeeze(0)  # [4]
# cos should be >= 0.8
print("cos(min/mean):", float(cos.min()), float(cos.mean()))

# --- cell separator ---

import torch
from collections import defaultdict

import torch
import torch.nn.functional as F
from typing import Optional, Literal
from transformers import CLIPTokenizerFast, CLIPTextModel

# ---------- cached CLIP text pieces ----------
_tok = CLIPTokenizerFast.from_pretrained("openai/clip-vit-large-patch14")
_txt = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14").eval()

@torch.no_grad()
def _tokens_and_hidden(text: str):
    """
    Returns:
      enc: HF tokenizer outputs
      toks: list[str] length 77
      hid: FloatTensor [1,77,768]
    """
    enc = _tok(
        text,
        return_tensors="pt",
        padding="max_length",
        max_length=77,
        truncation=True,
        add_special_tokens=True,
    )
    out = _txt(**enc)
    toks = _tok.convert_ids_to_tokens(enc["input_ids"][0].tolist())
    hid  = out.last_hidden_state  # [1,77,768]
    return enc, toks, hid

def _find_subseq(hay, sub):
    for i in range(len(hay) - len(sub) + 1):
        if hay[i:i+len(sub)] == sub:
            return i
    return -1

# ---------- unit-sphere sampler near a base direction ----------
@torch.no_grad()
def sample_token_nearby(
    base_hat: torch.Tensor,                      # [D] unit vector
    *,
    cosine_distance: float,                      # radius on cosine scale (0..2; typical 0..1)
    device: torch.device | str,
    dtype: torch.dtype,
    mode: Literal["uniform","threshold","center"] = "uniform",
    seed: Optional[int] = None,
) -> torch.Tensor:
    """
    Sample a unit vector y_hat with cosine(base_hat, y_hat) >= tau,
    where tau = 1 - cosine_distance. Returns [D] (unit).
    """
    base_hat = F.normalize(base_hat, dim=0).to(device=device, dtype=dtype)
    D = base_hat.numel()

    # rng
    if seed is not None:
        g = torch.Generator(device=device).manual_seed(int(seed))
    else:
        g = None

    tau = max(-1.0, min(1.0, 1.0 - float(cosine_distance)))  # clamp

    # choose cosine
    if mode == "center":
        c = torch.tensor(1.0, device=device, dtype=dtype)
    elif mode == "threshold":
        c = torch.tensor(tau, device=device, dtype=dtype)
    elif mode == "uniform":
        u = torch.rand((), generator=g, device=device, dtype=dtype)
        c = tau + (1 - tau) * u
    else:
        raise ValueError("mode must be one of {'uniform','threshold','center'}")

    # random orthogonal direction
    gvec = torch.randn(D, generator=g, device=device, dtype=dtype)
    proj = (gvec * base_hat).sum() * base_hat
    v = gvec - proj
    if torch.allclose(v.norm(), torch.tensor(0.0, device=device, dtype=dtype)):
        # extremely unlikely; resample
        v = torch.randn(D, generator=g, device=device, dtype=dtype)
        proj = (v * base_hat).sum() * base_hat
        v = v - proj
    v_hat = F.normalize(v, dim=0)

    s = torch.sqrt(torch.clamp(1 - c * c, min=0.0))
    y_hat = c * base_hat + s * v_hat
    return F.normalize(y_hat, dim=0)

# ---- replace with a *nearby sampled* embedding ----
@torch.no_grad()
def replace_token_with_nearby_embedding(
    prompt: str,
    search_word: str,
    *,
    device: torch.device | str = "cpu",
    cosine_distance: float = 0.2,           # “radius” on cosine scale
    combine: str = "mean_to_single",        # broadcast strategy across span
    replace_eos: bool = True,               # optionally overwrite EOS slots
    sample_mode: Literal["uniform","threshold","center"] = "uniform",
    seed: Optional[int] = None,
):
    """
    Find `search_word` in `prompt`, compute its contextual hidden span from CLIP,
    sample a *nearby* vector (within `cosine_distance`) using `sample_token_nearby`,
    and splice it back (broadcast over the span if needed).

    Returns:
      cond_hidden: [1,77,768] modified hidden states (on `device`)
      enc_orig: tokenizer output dict for original prompt
      meta: info including span indices and the achieved cosine similarity
    """
    # Encode original prompt
    enc_orig, toks_orig, hid_orig = _tokens_and_hidden(prompt)

    # Locate the BPE span for the word
    needle_src = _tok.tokenize(" " + search_word)
    if not needle_src:
        raise ValueError(f"search_word produced empty BPE sequence: {search_word!r}")
    L_src = len(needle_src)
    i_src = _find_subseq(toks_orig, needle_src)
    if i_src == -1:
        raise ValueError(f"Could not find BPE sequence for {search_word!r} in prompt tokens.")

    # Source span hidden states
    span_src = hid_orig[:, i_src:i_src + L_src, :]             # [1,Ls,768]
    # Use the mean as the base token vector to perturb (preserves context)
    src_mean = span_src.mean(dim=1).squeeze(0)                  # [768]

    # Sample a nearby *unit* vector, then match the original magnitude
    src_norm = float(src_mean.norm().clamp(min=1e-12).item())
    base_hat = src_mean / src_norm
    y_hat = sample_token_nearby(
        base_hat,
        cosine_distance=cosine_distance,
        device=device,
        dtype=src_mean.dtype,
        mode=sample_mode,
        seed=seed,
    )                                                           # [768] unit
    rep_vec = y_hat * src_norm                                  # [768] scaled like src_mean

    # Map to the span (either mean->single broadcast, or 1:1 if asked)
    if combine == "mean_to_single":
        rep_mapped = rep_vec.view(1, 1, -1).expand(1, L_src, -1)    # [1,Ls,768]
    else:  # "match_or_broadcast" (single vector -> broadcast)
        rep_mapped = rep_vec.view(1, 1, -1).expand(1, L_src, -1)

    # Replace in hidden states
    cond_hidden = hid_orig.clone()
    cond_hidden[:, i_src:i_src + L_src, :] = rep_mapped
    cond_hidden = cond_hidden.to(device)

    # Optional EOS replacement (use rep_vec)
    eos_id = _tok.eos_token_id
    ids_orig = enc_orig["input_ids"][0]                          # [77]
    span_ids = ids_orig[i_src:i_src + L_src]
    non_eos_replaced = int((span_ids != eos_id).sum().item())

    if replace_eos:
        rep_eos = rep_vec.view(1, 1, -1).to(device)              # [1,1,768]
        eos_positions_orig = torch.nonzero(ids_orig == eos_id, as_tuple=False).flatten()
        if eos_positions_orig.numel() > 0:
            # bring index tensor to CPU for advanced indexing if needed
            idx = eos_positions_orig.to(torch.long)
            cond_hidden[:, idx, :] = rep_eos.expand(1, idx.numel(), -1)

    # Report achieved cosine (w.r.t. original mean direction)
    base_hat = base_hat.to(device)
    cos_achieved = float(F.cosine_similarity(base_hat, F.normalize(rep_vec.to(device), dim=0), dim=0).item())

    meta = {
        "prompt": prompt,
        "search_bpe": needle_src,
        "src_start": i_src,
        "src_len": L_src,
        "cosine_distance_req": float(cosine_distance),
        "cosine_similarity_achieved": cos_achieved,
        "non_eos_replaced_in_span": non_eos_replaced,
        "sample_mode": sample_mode,
        "seed": seed,
    }
    return cond_hidden, enc_orig, meta

# --- cell separator ---

import os, random
def set_global_seed(seed: int = 2):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # (optional) more determinism
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    # torch.use_deterministic_algorithms(True)  # enable if you want strict determinism

# 1) set seeds


if __name__ == "__main__":
    set_global_seed(5)

    prompt = "a photo of the truck"
    cond_replaced, _, meta = replace_token_with_nearby_embedding(
        prompt,
        search_word="truck",
        sample_mode="uniform",
        replace_eos=True,
        cosine_distance=1.0,
        seed=71995,
        device=model.device
    )
    print(cond_replaced.shape)
    i0, L = meta["src_start"], meta["src_len"]
    indices = list(range(i0, i0 + L))
    print("Replaced token indices in 77-length seq:", indices)
    # For Stable Diffusion (CompVis LDM style), you can pass:
    # conditioning={"c_crossattn": [cond_hidden]}  (instead of model.get_learned_conditioning([...]))
    # Unconditional stays the usual empty prompt:
    # uncond = model.get_learned_conditioning([""])

    orig = model.get_learned_conditioning([prompt]).to(model.device)
    print("Δcond L2:", (cond_replaced - orig).norm().item())
    print("cosine_similarity_achieved", meta["cosine_similarity_achieved"])

    batch_size = 1
    device = model.device
    start_code = torch.randn(batch_size, 4, 64, 64, device=device)  # 512x512
    cond = model.get_learned_conditioning([prompt] * start_code.shape[0])
    pokemon = model.get_learned_conditioning(['pokemon'] * start_code.shape[0])

    with torch.no_grad(), torch.autocast(device_type="cuda", enabled=False):
        generated_img = generate_and_save_sd_images(
            model=model,
            sampler=sampler,
            cond=cond_replaced,
            device=torch.device("cuda"),
            steps=50,
            out_dir="tmp",
            start_code=start_code,
            prefix="orig_",
        )

        print('!! ', generated_img.shape)  # [0,1]

    # --- cell separator ---

    x = generated_img[0]  # [3,H,W] or [H,W,3], torch.Tensor or np.ndarray

    pass
