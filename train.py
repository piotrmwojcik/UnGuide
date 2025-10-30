#!/usr/bin/env python3
import argparse
import json
import sys
import random
import os
from functools import partial
from pathlib import Path

import matplotlib.pyplot as plt  # (unused, but kept if you uncomment stats)
import numpy as np
import torch
import wandb

from typing import Iterator, Tuple, Dict, Any, Union, Callable

from typing import Dict, Any, Iterator, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, List, Tuple

# assumes you have these classes
from hyper_lora import HyperLora, HyperLoRALinear

# ---- helpers ----
from typing import Iterator, Tuple, Dict, Any, List, Optional
import torch
import torch.nn as nn

# import your classes
from hyper_lora import HyperLora, HyperLoRALinear
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed as hf_set_seed
from torch.utils.data import DataLoader
import pandas as pd
from torchvision.transforms.functional import to_pil_image
from transformers import CLIPTextModel, CLIPTokenizer
from tqdm import tqdm

from data_utils import TargetReferenceDataset, collate_prompts
from ldm.models.diffusion.ddimcopy import DDIMSampler
from ldm.util import instantiate_from_config  # (kept if your get_models uses it)
from hyper_lora import (HyperLoRALinear, HypernetworkManager, inject_hyper_lora,
                        inject_hyper_lora_nsfw)
from sampling import sample_model
from utils import get_models, print_trainable_parameters  # DO NOT import set_seed here to avoid clashes


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="LoRA/HyperLoRA Fine-tuning for Stable Diffusion (Accelerate)"
    )

    # Model configuration
    parser.add_argument(
        "--config_path",
        type=str,
        default="./configs/stable-diffusion/v1-inference.yaml",
        help="Path to model configuration file",
    )
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default="./models/sd-v1-4.ckpt",
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--device", type=str, default="cuda:0",
        help="(Ignored when using Accelerate) Device to use for training"
    )

    parser.add_argument(
        "--target_object",
        type=str,
        default="ship",
        help="Target concept to be removed.",
    )

    # LoRA/HyperLoRA
    parser.add_argument("--lora_rank", type=int, default=1, help="LoRA rank parameter")
    parser.add_argument("--lora_alpha", type=float, default=8, help="LoRA alpha parameter")
    parser.add_argument(
        "--target_modules",
        nargs="+",
        default=["attn2.to_k", "attn2.to_v"],
        help="Target modules for LoRA injection",
    )
    parser.add_argument("--clip_size", type=int, default=768, help="CLIP embedding size")

    # Optim/Trainer
    parser.add_argument("--iterations", type=int, default=200, help="Number of training iterations")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--lr", type=float, default=3e-5, help="Learning rate")
    parser.add_argument("--image_size", type=int, default=512, help="Image size for training")
    parser.add_argument("--ddim_steps", type=int, default=50, help="DDIM sampling steps")
    parser.add_argument("--ddim_eta", type=float, default=0.0, help="DDIM eta")
    parser.add_argument("--start_guidance", type=float, default=9.0, help="Starting guidance scale")
    parser.add_argument("--negative_guidance", type=float, default=2.0, help="Negative guidance scale")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--internal_lr", type=float, default=1e-4, help="Simulated lr for hypernetwork")
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument("--csv_path", type=str, default=None, help="csv")
    parser.add_argument("--use_dummy_embeddings", action="store_true", help="Use dummy zero tensors instead of loading embeddings (for testing)")

    # Logging / tracking
    parser.add_argument("--use-wandb", action="store_true", dest="use_wandb")
    parser.add_argument("--log_from", type=int, default=0, help="Log debug images from iteration")
    parser.add_argument(
        "--logging_dir", type=str, default="logs",
        help="Base logging directory (used by Accelerate trackers)."
    )
    parser.add_argument(
        "--mixed_precision", type=str, default=None, choices=["no", "fp16", "bf16"],
        help="Override Accelerate mixed precision (fp16/bf16)."
    )
    parser.add_argument(
        "--report_to", type=str, default="wandb",
        help='Tracking integration: "tensorboard", "wandb", "comet_ml", or "all".'
    )

    # Output / data
    parser.add_argument("--output_dir", type=str, default="output", help="Directory to save models")
    parser.add_argument("--data_dir", type=str, default="data10", help="Directory with prompt json files")
    parser.add_argument("--save_losses", action="store_true", help="Save training losses to file")

    return parser.parse_args()


CIFAR100 = [
    'apple','aquarium fish','baby','bear','beaver','bed','bee','beetle','bicycle','bottle',
    'bowl','boy','bridge','bus','butterfly','camel','can','castle','caterpillar','cattle',
    'chair','chimpanzee','clock','cloud','cockroach','couch','crab','crocodile','cup','dinosaur',
    'dolphin','elephant','flatfish','forest','fox','girl','hamster','house','kangaroo','keyboard',
    'lamp','lawn mower','leopard','lion','lizard','lobster','man','maple tree','motorcycle','mountain',
    'mouse','mushroom','oak tree','orange','orchid','otter','palm tree','pear','pine tree',
    'plain','plate','poppy','porcupine','possum','rabbit','raccoon','ray','road','rocket',
    'rose','sea','seal','shark','shrew','skunk','skyscraper','snail','snake','spider',
    'squirrel','streetcar','sunflower','sweet pepper','table','tank','telephone','television','tiger','tractor',
    'train','trout','tulip','turtle','wardrobe','whale','willow tree','wolf','woman','worm'
]

def prompt_augmentation(content, augment=True):
    if augment:
        prompts = [
            # object augmentation
            "{} in a photo".format(content),
            "{} in a snapshot".format(content),
            "A snapshot of {}".format(content),
            "A photograph showcasing {}".format(content),
            "An illustration of {}".format(content),
            "A digital rendering of {}".format(content),
            "A visual representation of {}".format(content),
            "A graphic of {}".format(content),
            "A shot of {}".format(content),
            "A photo of {}".format(content),
            "A black and white image of {}".format(content),
            "A depiction in portrait form of {}".format(content),
            "A scene depicting {} during a public gathering".format(content),
            "{} captured in an image".format(content),
            "A depiction created with oil paints capturing {}".format(content),
            "An image of {}".format(content),
            "A drawing capturing the essence of {}".format(content),
            "An official photograph featuring {}".format(content),
            "A detailed sketch of {}".format(content),
            "{} during sunset/sunrise".format(content),
            "{} in a detailed portrait".format(content),
            "An official photo of {}".format(content),
            "Historic photo of {}".format(content),
            "Detailed portrait of {}".format(content),
            "A painting of {}".format(content),
            "HD picture of {}".format(content),
            "Magazine cover capturing {}".format(content),
            "Painting-like image of {}".format(content),
            "Hand-drawn art of {}".format(content),
            "An oil portrait of {}".format(content),
            "{} in a sketch painting".format(content),
        ]

        return prompts

#CIFAR100 = ['castle', 'apple']


def create_quick_sampler(model, sampler, image_size: int, ddim_steps: int, ddim_eta: float):
    """Create a quick sampling function with fixed parameters"""
    return lambda conditioning, scale, start_code, till_T: sample_model(
        model,
        sampler,
        conditioning,
        image_size,
        image_size,
        ddim_steps,
        scale,
        ddim_eta,
        start_code=start_code,
        till_T=till_T,
        verbose=False,
    )

_KEY_PAIRS = [
    ("alpha_grad",         "alpha"),
    ("x_L_grad",           "x_L"),
    ("x_R_grad",           "x_R"),
]

_LIVE_GETTERS = [
    ("alpha",          lambda hl: hl.alpha if getattr(hl, "use_scaling", False) and hasattr(hl, "alpha") else None),
    ("x_L",            lambda hl: getattr(hl, "_last_x_L", None)),
    ("x_R",            lambda hl: getattr(hl, "_last_x_R", None)),
]


def flatten_live_tensors(model_wrapped: nn.Module, accelerator) -> torch.Tensor:
    """
m    This preserves autograd so loss can backprop through them.
    """
    base = accelerator.unwrap_model(model_wrapped)
    parts: List[torch.Tensor] = []

    for _, hl in _iter_hyperlora_layers(base):
        for _, getter in _LIVE_GETTERS:
            t = getter(hl)
            if t is None:
                continue
            # ensure Tensor, keep graph
            parts.append(t.view(-1))

    if not parts:
        return torch.tensor([], device=next(base.parameters()).device)

    return torch.cat(parts, dim=0)


def _concat_items(
    items: List[Tuple[str, str, torch.Tensor]],
    device: Optional[torch.device],
    dtype: Optional[torch.dtype],
) -> Tuple[torch.Tensor, List[Dict[str, Any]]]:
    parts: List[torch.Tensor] = []
    index: List[Dict[str, Any]] = []
    n = 0
    for layer, key, t in items:
        vec = t.detach()
        if device is not None:
            vec = vec.to(device)
        if dtype is not None and vec.dtype != dtype:
            vec = vec.to(dtype)
        vec = vec.view(-1)
        start, end = n, n + vec.numel()
        parts.append(vec)
        index.append({
            "layer": layer,
            "key": key,
            "shape": tuple(t.shape),
            "start": start,
            "end": end,
        })
        n = end

    if parts:
        flat = torch.cat(parts, dim=0)
    else:
        flat = torch.empty(0, device=device or torch.device("cpu"),
                           dtype=dtype or torch.float32)
    return flat, index


def concat_grads_and_tensors(
    recs: Dict[str, Dict[str, Any]],
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = torch.float32,
) -> Dict[str, Any]:
    """
    recs: dict from collect_hyperlora_tensors_and_grads(...)
    returns:
      {
        'grads_flat':   1D tensor,
        'grads_index':  [{layer,key,shape,start,end}, ...],
        'tensors_flat': 1D tensor,
        'tensors_index':[{layer,key,shape,start,end}, ...],
      }
    Ordering is stable: by layer name (sorted), then by _KEY_PAIRS.
    Only non-None entries are included.
    """
    grad_items: List[Tuple[str, str, torch.Tensor]] = []
    tensor_items: List[Tuple[str, str, torch.Tensor]] = []

    for layer in sorted(recs.keys()):
        rec = recs[layer]
        for gk, vk in _KEY_PAIRS:
            g = rec.get(gk, None)
            v = rec.get(vk, None)
            if g is not None:
                grad_items.append((layer, gk, g))
            if v is not None:
                tensor_items.append((layer, vk, v))

    grads_flat, grads_index = _concat_items(grad_items, device, dtype)
    tensors_flat, tensors_index = _concat_items(tensor_items, device, dtype)

    return {
        "grads_flat": grads_flat,
        "grads_index": grads_index,
        "tensors_flat": tensors_flat,
        "tensors_index": tensors_index,
    }


def generate_and_save_sd_images(
    model,
    sampler,
    prompt: str,
    device: torch.device,
    steps: int = 50,
    eta: float = 0.0,
    batch_size: int = 1,
    out_dir: str = "tmp",
    prefix: str = "unl_",
    start_code: torch.Tensor = None,   # optional noise tensor [B,4,64,64] for 512x512
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
    - start_code: optional start noise shape [B, 4, H/8, W/8]; if None, sampled internally.
                  For 512×512 set shape to [B, 4, 64, 64].
    """
    if start_code is None:
        start_code = torch.randn(batch_size, 4, 64, 64, device=device)  # 512x512

    model.eval()
    with torch.no_grad(), torch.autocast(device_type=device.type, enabled=(device.type == "cuda")):
        cond   = model.get_learned_conditioning([prompt] * start_code.shape[0])
        uncond = model.get_learned_conditioning([""] * start_code.shape[0])

        samples_latent, _ = sampler.sample(
            S=steps,
            conditioning={"c_crossattn": [cond]},
            batch_size=start_code.shape[0],
            shape=start_code.shape[1:],  # (4, H/8, W/8)
            verbose=False,
            unconditional_guidance_scale=7.5,
            unconditional_conditioning={"c_crossattn": [uncond]},
            eta=eta,
            x_T=start_code,
        )

        imgs = model.decode_first_stage(samples_latent)       # [-1, 1]
        imgs = (imgs.clamp(-1, 1) + 1) / 2.0                 # [0, 1]

        out_path = Path(out_dir)
        out_path.mkdir(exist_ok=True, parents=True)
        for i, im in enumerate(imgs.cpu()):
            im_u8 = (im.clamp(0, 1) * 255).round().to(torch.uint8)  # [3,H,W]
            to_pil_image(im_u8).save(out_path / f"{prefix}{i:04d}.png")

        return imgs  # [B,3,H,W] in [0,1]


def _iter_hyperlora_layers(root: nn.Module):
    seen = set()
    for name, m in root.named_modules():
        if isinstance(m, HyperLoRALinear):
            hl = m.hyper_lora
            if id(hl) in seen:
                continue
            seen.add(id(hl))
            yield name + ".hyper_lora", hl
        elif isinstance(m, HyperLora):
            if id(m) in seen:
                continue
            seen.add(id(m))
            yield name, m


# def _iter_hyperlora_layers(root: nn.Module) -> Iterator[Tuple[str, HyperLora]]:
#     """Yield (qualified_name, HyperLora) whether wrapped or direct."""
#     for name, m in root.named_modules():
#         if isinstance(m, HyperLoRALinear):
#             #print('--hyperloralinera---')
#             yield name + ".hyper_lora", m.hyper_lora
#         elif isinstance(m, HyperLora):
#             #print('--hyperlinera---')
#             yield name, m


def pooled_from_hidden_and_prompt(
        last_hidden: torch.Tensor,  # [77, 768] or [1,77,768]
        prompt: str,
        tokenizer,  # optional; from openai/clip-vit-large-patch14
):
    """
    Returns:
      pooled: [768] if clip_model is None; else the projected & L2-normed feature (e.g. [512])
      eos_idx: int, the EOS position we used
    """
    # Make shapes consistent
    if last_hidden.ndim == 2:  # [77,768] -> [1,77,768]
        last_hidden = last_hidden.unsqueeze(0)
    assert last_hidden.shape[1] == 77, "Expected sequence length 77"

    # Tokenize prompt to get attention mask (EOS = last non-pad token)
    enc = tokenizer(
        prompt,
        return_tensors="pt",
        padding="max_length",
        max_length=77,
        truncation=True,
        add_special_tokens=True,
    )
    # eos index is the last position with mask==1
    eos_idx = int(enc["attention_mask"].sum(dim=-1).item() - 1)

    # Take the EOS hidden state
    pooled = last_hidden[0, eos_idx, :]  # [768]

    return pooled, eos_idx


def collect_hyperlora_tensors_and_grads(
    model_wrapped: nn.Module, accelerator, verbose: bool = False, all_ranks: bool = True
) -> Dict[str, Dict[str, Any]]:
    """
    Collect only alpha, x_L, x_R and their grads.
    Prints per-layer grad norms and warns if any is None.
    """
    base = accelerator.unwrap_model(model_wrapped)
    out: Dict[str, Dict[str, Any]] = {}

    # decide printing
    rank = int(os.environ.get("RANK", "0"))
    do_print = (accelerator.is_main_process or all_ranks)

    def _say(msg: str):
        if verbose and do_print:
            print(f"[RANK {rank}] {msg}", flush=True)

    for lname, hl in _iter_hyperlora_layers(base):
        rec: Dict[str, Any] = {}

        # values (detached copies for logging/saving)
        if getattr(hl, "use_scaling", False) and hasattr(hl, "alpha"):
            rec["alpha"] = hl.alpha.detach().clone()
        else:
            rec["alpha"] = None

        xL = getattr(hl, "_last_x_L", None)
        xR = getattr(hl, "_last_x_R", None)
        rec["x_L"] = None if xL is None else xL.detach().clone()
        rec["x_R"] = None if xR is None else xR.detach().clone()

        # grads (detached copies)
        if getattr(hl, "use_scaling", False) and hasattr(hl, "alpha"):
            a_g = hl.alpha.grad
            if a_g is None:
                _say(f"HyperLoRA:{lname} — NO GRAD for alpha (alpha.grad is None)")
                rec["alpha_grad"] = None
            else:
                rec["alpha_grad"] = a_g.detach().clone()
        else:
            _say(f"HyperLoRA:{lname} — alpha not used (use_scaling=False or missing)")
            rec["alpha_grad"] = None

        if xL is None:
            _say(f"HyperLoRA:{lname} — NO TENSOR x_L (_last_x_L not set; retain_grad()/forward?)")
            rec["x_L_grad"] = None
        else:
            if xL.grad is None:
                _say(f"HyperLoRA:{lname} — NO GRAD for x_L (x_L.grad is None)")
                rec["x_L_grad"] = None
            else:
                rec["x_L_grad"] = xL.grad.detach().clone()

        if xR is None:
            _say(f"HyperLoRA:{lname} — NO TENSOR x_R (_last_x_R not set; retain_grad()/forward?)")
            rec["x_R_grad"] = None
        else:
            if xR.grad is None:
                _say(f"HyperLoRA:{lname} — NO GRAD for x_R (x_R.grad is None)")
                rec["x_R_grad"] = None
            else:
                rec["x_R_grad"] = xR.grad.detach().clone()

        # always print summary norms so you know the function ran
        def _n(t):
            return "None" if t is None else f"{t.detach().norm().item():.3e}"
        _say(
            f"HyperLoRA:{lname} | dα={_n(rec['alpha_grad'])} "
            f"| ||d x_L||={_n(rec['x_L_grad'])} | ||d x_R||={_n(rec['x_R_grad'])}"
        )

        out[lname] = rec

    return out


import torch
import torch.nn.functional as F
from typing import Optional, Literal


def main():
    args = parse_args()

    # Print basic config
    print("=== LoRA/HyperLoRA Fine-tuning (Accelerate) ===")
    print(f"Config: {args.config_path}")
    print(f"Checkpoint: {args.ckpt_path}")
    print(f"Output dir: {args.output_dir}")
    print(f"Iterations: {args.iterations}  |  LR: {args.lr}  |  Accum: {args.gradient_accumulation_steps}")
    print(f"Image size: {args.image_size}  |  DDIM steps: {args.ddim_steps}  |  eta: {args.ddim_eta}")
    print("=" * 48)

    # Seed
    if args.seed is not None:
        hf_set_seed(args.seed)

    # Accelerate project config
    accelerator_project_config = ProjectConfiguration(
        project_dir=args.output_dir,
        logging_dir=args.logging_dir,
    )

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,  # None -> use accelerate config
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )

    args.lr = (
            args.lr
            * args.gradient_accumulation_steps
            * args.batch_size
            * accelerator.num_processes
    )

    df = pd.read_csv(args.csv_path)
    THRESHOLD_RETAIN = 0.025
    THRESHOLD_REMOVE = 0.015
    for col in ["clip_cos_replaced", "clip_cos_baseline"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Compute the gap and filter
    df["clip_gap"] = df["clip_cos_baseline"] - df["clip_cos_replaced"]
    retain_prompts = df[df["clip_gap"] >= THRESHOLD_RETAIN]
    remove_prompts = df[(df["clip_gap"] < 0) | (df["clip_gap"] < THRESHOLD_REMOVE)]

    def load_tensors(paths):
        tensors, ok_paths = [], []
        for p in paths:
            try:
                t = torch.load(p, map_location="cpu")
                tensors.append(t)
                ok_paths.append(p)
            except FileNotFoundError:
                print(f"[missing] {p}")
        return tensors, ok_paths

    EMB_ROOT = Path(f"random_replacements_{args.target_object}_emb")  # folder with the .pt files
    def rows_to_paths(df_sub, root=EMB_ROOT):
        paths = []
        for _, r in df_sub.iterrows():
            i = int(r["idx"])
            s = int(r["seed"])
            p = root / f"idx{i:03d}_seed{s}_spanmean.pt"
            paths.append(p)
        return paths

    retain_paths = rows_to_paths(retain_prompts)
    remove_paths = rows_to_paths(remove_prompts)

    if args.use_dummy_embeddings:
        print("WARNING: Using dummy zero tensors for embeddings (testing mode)")
        dummy_embedding = torch.zeros(77, 768)
        retain_tensors = [dummy_embedding for _ in range(max(1, len(retain_prompts)))]
        remove_tensors = [dummy_embedding for _ in range(max(1, len(remove_prompts)))]
    else:
        retain_tensors, _ = load_tensors(retain_paths)
        remove_tensors, _ = load_tensors(remove_paths)

        if not retain_tensors or not remove_tensors:
            raise ValueError(
                f"Failed to load embeddings. "
                f"Found {len(retain_tensors)} retain tensors and {len(remove_tensors)} remove tensors. "
                f"Expected embeddings in: {EMB_ROOT}. "
                f"Please generate the embeddings first, use --use_dummy_embeddings flag for testing, "
                f"or provide a valid csv_path."
            )

    #logger = get_logger(__name__)
    is_main = accelerator.is_main_process

    # Trackers (W&B/TB/etc.) — initialize after Accelerator so it attaches run metadata
    if is_main and args.use_wandb and ("wandb" in str(args.report_to) or args.report_to == "all"):
        wandb.init(project="UnGuide", name="training", config=vars(args))

    # Data
    data_dir = args.data_dir
    overs = {f"{args.target_object}.json": 8.0, "neutral.json": 4.0}
    ds = TargetReferenceDataset(data_dir, weights=overs)
    ds_loader = DataLoader(ds, batch_size=1, shuffle=True, collate_fn=collate_prompts)

    # Models (original + trainable clone)
    if is_main:
        os.makedirs(os.path.join(args.output_dir, "tmp"), exist_ok=True)

    model_orig, sampler_orig, model, sampler_unused = get_models(
        args.config_path, args.ckpt_path, accelerator.device
    )

    # Freeze original model
    for p in model_orig.model.diffusion_model.parameters():
        p.requires_grad = False
    model_orig.eval()

    for p in model.model.diffusion_model.parameters():
        p.requires_grad = False

    model.hyper = HypernetworkManager()

    # Inject HyperLoRA/LoRA BEFORE prepare(), then build optimizer on trainable params
    use_hyper = True  # your script forces hypernetwork on; keep same behavior
    hyper_lora_factory = partial(
        HyperLoRALinear,
        clip_size=args.clip_size,
        rank=args.lora_rank,
        alpha=args.lora_alpha,
    )
    hyper_lora_layers = inject_hyper_lora(
        model.model.diffusion_model, args.target_modules, hyper_lora_factory
    )
    for layer_name, layer in hyper_lora_layers:
        layer.set_parent_model(model)

    # Optimizer on trainable (LoRA) params only
    trainable_params = list(filter(lambda p: p.requires_grad, model.model.diffusion_model.parameters()))
    if is_main:
        print(f"Total trainable parameter tensors: {len(trainable_params)}")
        print_trainable_parameters(model)

    optimizer = torch.optim.Adam(trainable_params, lr=args.lr)

    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[300], gamma=0.5
    )

    # Prepare for DDP / Mixed precision
    model, optimizer, ds_loader = accelerator.prepare(model, optimizer, ds_loader)

    for layer_name, layer in hyper_lora_layers:
        layer.set_parent_model(accelerator.unwrap_model(model))
        base.hyper.add_hyperlora(layer_name, layer.hyper_lora)

        # Create sampler AFTER prepare so it uses the wrapped model
    sampler = DDIMSampler(base)
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
    # Tokenizer + CLIP text encoder (inference-only; keep unwrapped)
    clip_text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14").to(accelerator.device).eval()

    # Quick sampler
    quick_sampler = create_quick_sampler(base, sampler, args.image_size, args.ddim_steps, args.ddim_eta)

    def encode(text: str):
        return (
            tokenizer(
                text,
                max_length=tokenizer.model_max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
                .to(accelerator.device)
                .input_ids
        )

    # Training
    criterion = torch.nn.MSELoss()
    losses = []

    pbar = tqdm(range(args.iterations), disable=not accelerator.is_local_main_process)
    for i in pbar:
        for sample_ids, sample in enumerate(ds_loader):
            target_text = f"a photo of the {args.target_object}"

            # Get conditional embeddings (strings) directly for LDM
            emb_0 = base.get_learned_conditioning(sample["reference"])
            emb_p = base.get_learned_conditioning(sample["target"])
            emb_n = base.get_learned_conditioning(sample["target"])

            optimizer.zero_grad(set_to_none=True)

            # random timestep mapping (keep your logic)
            t_enc = torch.randint(args.ddim_steps, (1,), device=accelerator.device)
            og_num = round((int(t_enc) / args.ddim_steps) * 1000)
            og_num_lim = round((int(t_enc + 1) / args.ddim_steps) * 1000)
            t_enc_ddpm = torch.randint(og_num, og_num_lim, (1,), device=accelerator.device)

            # Build CLIP tokens for current target/reference (for HyperLoRA conditioning)

            #inputs = encode(sample["target"])
            #print('!!! ', sample["target"])
            inputs_other = encode("a photo of the bird")
            inputs_other2 = encode("a photo of the dog")
            inputs_other3 = encode("a photo of the feline")
            inputs_target = encode(target_text)
            with torch.no_grad():
                #cond_target = clip_text_encoder(inputs).pooler_output.detach()
                cond_other = clip_text_encoder(inputs_other).pooler_output.detach()
                cond_other2 = clip_text_encoder(inputs_other2).pooler_output.detach()
                cond_other3 = clip_text_encoder(inputs_other3).pooler_output.detach()
                cond_target = clip_text_encoder(inputs_target).pooler_output.detach()

                #cond_ref    = clip_text_encoder(inputs[1]).pooler_output.detach()

            with torch.no_grad():
                remove_prompt = random.choice(remove_tensors).detach()
                remove_prompt, _ = pooled_from_hidden_and_prompt(remove_prompt, target_text,
                                                                tokenizer=tokenizer)
                remove_prompt = remove_prompt.unsqueeze(dim=0).to(base.device)
            base.hyper.set_context(remove_prompt, int(torch.randint(0, 149, (1,), device=accelerator.device)))
            # starting latent code
            start_code = torch.randn(
                (1, 4, args.image_size // 8, args.image_size // 8),
                device=accelerator.device
            )
            with accelerator.accumulate(model):
                if 'neutral.json' in sample['file']:
                    with torch.no_grad():
                        retain_prompt = random.choice(retain_tensors)
                        retain_prompt, _ = pooled_from_hidden_and_prompt(retain_prompt, target_text,
                                                                         tokenizer=tokenizer)
                        retain_prompt = retain_prompt.unsqueeze(dim=0).to(base.device).detach()
                        accelerator.unwrap_model(model).hyper.set_context(retain_prompt, 0)

                    accelerator.unwrap_model(model).hyper.compute_and_cache_loras(retain_prompt, 0)
                    tensors_flat_t_live = flatten_live_tensors(model, accelerator)
                    t_ = int(torch.randint(1, 150, (1,), device=accelerator.device))
                    accelerator.unwrap_model(model).hyper.set_context(retain_prompt, 150)
                    accelerator.unwrap_model(model).hyper.compute_and_cache_loras(retain_prompt, 150)
                    #tensors_flat_t1_live = flatten_live_tensors(model, accelerator)
                    #print('!!! ', tensors_flat_t1_live.shape, tensors_flat_t1_live - tensors_flat_t_live)
                    #delta_live = tensors_flat_t1_live - tensors_flat_t_live
                    loss = (tensors_flat_t_live ** 2).mean()

                    loss_for_backward = loss / accelerator.gradient_accumulation_steps
                    print('loss neutral ', loss_for_backward)
                else:
                    with torch.no_grad():
                        z = quick_sampler(emb_p, args.start_guidance, start_code, int(t_enc))
                        e_0 = model_orig.apply_model(z, t_enc_ddpm, emb_0)  # reference (stopgrad)
                        e_p = model_orig.apply_model(z, t_enc_ddpm, emb_p)  # target   (stopgrad)

                    # prediction for trainable model (needs grads)
                    e_n = base.apply_model(z, t_enc_ddpm, emb_n)

                    # targets and loss
                    e_0.requires_grad_(False)
                    e_p.requires_grad_(False)
                    target = e_0 - (args.negative_guidance * (e_p - e_0))
                    loss = criterion(e_n, target)  # per-rank scalar tensor

                    # ---- BACKWARD (per micro-step) ----
                    # Scale the loss for gradient accumulation so the effective grad equals the true average
                    loss_for_backward = loss / accelerator.gradient_accumulation_steps
                    accelerator.backward(loss_for_backward, retain_graph=True)
                    dev = next(accelerator.unwrap_model(model).parameters()).device

                    recs = collect_hyperlora_tensors_and_grads(model, accelerator)
                    pack = concat_grads_and_tensors(recs, device=dev)  # has 'grads_flat', 'tensors_flat', etc.

                    # Target step: Δθ ≈ -lr * g_t  (keep target detached)
                    grads_flat_t = -1 * args.internal_lr * pack["grads_flat"].detach()

                    # --- LIVE anchor at t (no detach!) ---
                    tensors_flat_t_live = flatten_live_tensors(model, accelerator)
                    #print('!! ', pack["grads_flat"].shape, pack["tensors_flat"].shape, tensors_flat_t_live.shape)

                    # Clear grads before the next forward
                    #optimizer.zero_grad(set_to_none=True)


                    _, current_timestep = accelerator.unwrap_model(model).hyper.get_context()
                    base.hyper.set_context(remove_prompt, current_timestep + 1)
                    _ = base.apply_model(z, t_enc_ddpm, emb_n)

                    # LIVE vector at t+1 (keeps graph)
                    tensors_flat_t1_live = flatten_live_tensors(model, accelerator)

                    # Match the SGD step: (θ_{t+1} - θ_t) ≈ -lr * g_t
                    delta_live = tensors_flat_t1_live - tensors_flat_t_live

                    # e.g., MSE to the target step
                    loss = criterion(delta_live, grads_flat_t)
                    loss_for_backward = loss / accelerator.gradient_accumulation_steps
                    print('loss remove ', loss_for_backward)
                accelerator.backward(loss_for_backward)

                # ---- OPTIMIZER STEP (only on last micro-step) ----
                if accelerator.sync_gradients:
                    lrs_before = [pg["lr"] for pg in optimizer.param_groups]
                    accelerator.print(f"[iter {i}] LR before sched: " + ", ".join(f"{lr:.6e}" for lr in lrs_before))

                    # (optional) gradient clipping
                    # accelerator.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    scheduler.step()

                    lrs_after = [pg["lr"] for pg in optimizer.param_groups]
                    accelerator.print(f"[iter {i}] LR after sched: " + ", ".join(f"{lr:.6e}" for lr in lrs_after))
                    if accelerator.is_local_main_process and args.use_wandb:
                        assert len(lrs_after) == 1
                        wandb.log(
                            {"lr_group": lrs_after[0]},  # include your custom step metric if you use it in the UI
                            step=i
                        )
            # Optional image logging
            if (
                accelerator.is_local_main_process
                and args.use_wandb
                and i >= args.log_from
                and i % 10 == 0
                and sample_ids == 0
            ):
                base.hyper.set_context(cond_target, 150)
                imgs = generate_and_save_sd_images(
                    model=base,
                    sampler=sampler,
                    prompt=f"a photo of the {args.target_object}",
                    device=accelerator.device,
                    steps=50,
                    out_dir=os.path.join(args.output_dir, "tmp"),
                    prefix=f"unl_{i}_",
                )
                if imgs is not None:
                    caption = f"target: {args.target_object}"
                    im0 = (imgs[0].clamp(0, 1) * 255).round().to(torch.uint8).cpu()
                    wandb.log({"sample": wandb.Image(to_pil_image(im0), caption=caption)}, step=i)
                base.hyper.set_context(cond_other, 150)
                imgs = generate_and_save_sd_images(
                    model=base,
                    sampler=sampler,
                    prompt="a photo of the bird",
                    device=accelerator.device,
                    steps=50,
                    out_dir=os.path.join(args.output_dir, "tmp"),
                    prefix=f"unl_{i}_",
                )
                if imgs is not None:
                    caption = f"target: bird"
                    im0 = (imgs[0].clamp(0, 1) * 255).round().to(torch.uint8).cpu()
                    wandb.log({"sample (other)": wandb.Image(to_pil_image(im0), caption=caption)}, step=i)
                base.hyper.set_context(cond_other2, 150)
                imgs = generate_and_save_sd_images(
                    model=base,
                    sampler=sampler,
                    prompt="a photo of the dog",
                    device=accelerator.device,
                    steps=50,
                    out_dir=os.path.join(args.output_dir, "tmp"),
                    prefix=f"unl_{i}_",
                )
                if imgs is not None:
                    caption = f"target: dog"
                    im0 = (imgs[0].clamp(0, 1) * 255).round().to(torch.uint8).cpu()
                    wandb.log({"sample (other) 2": wandb.Image(to_pil_image(im0), caption=caption)}, step=i)
                base.hyper.set_context(cond_other3, 150)
                imgs = generate_and_save_sd_images(
                    model=base,
                    sampler=sampler,
                    prompt="a photo of the feline",
                    device=accelerator.device,
                    steps=50,
                    out_dir=os.path.join(args.output_dir, "tmp"),
                    prefix=f"unl_{i}_",
                )
                if imgs is not None:
                    caption = f"target: feline"
                    im0 = (imgs[0].clamp(0, 1) * 255).round().to(torch.uint8).cpu()
                    wandb.log({"sample (other) 3": wandb.Image(to_pil_image(im0), caption=caption)}, step=i)
            with torch.no_grad():
                loss_reduced = accelerator.gather(loss.detach()).mean()

            loss_value = float(loss_reduced.item())
            losses.append(loss_value)

            if accelerator.is_local_main_process and args.use_wandb:
                wandb.log({"loss": loss_value}, step=i)

            if accelerator.is_local_main_process:
                pbar.set_postfix({"loss": f"{loss_value:.6f}"})

        # Save LoRA/HyperLoRA weights each iteration (or move outside loop if you prefer)
        accelerator.wait_for_everyone()
        if accelerator.is_local_main_process:
            save_dir = os.path.join(args.output_dir, f"rank_{args.lora_rank}_it_{args.iterations}_lr_{args.lr}_sg_{args.start_guidance}_ng_{args.negative_guidance}_ddim_{args.ddim_steps}_" + ("hyper" if use_hyper else "lora"))
            os.makedirs(os.path.join(save_dir, "models"), exist_ok=True)

            lora_state_dict = {}
            # unwrap model for named_parameters if needed
            model_unwrapped = accelerator.unwrap_model(model)
            for name, param in model_unwrapped.model.diffusion_model.named_parameters():
                if param.requires_grad:
                    lora_state_dict[name] = param.detach().cpu().clone()

            model_filename = "hyper_lora.pth" if use_hyper else "lora.pth"
            lora_path = os.path.join(save_dir, "models", model_filename)
            accelerator.save(lora_state_dict, lora_path)

    # Wrap up
    if accelerator.is_local_main_process:
        print("Training completed!")
        if losses:
            print(f"Final loss: {losses[-1]:.6f}")
            print(f"Average loss: {sum(losses)/len(losses):.6f}")

        # Dump config + basic metrics
        run_dir = os.path.dirname(lora_path) if losses else args.output_dir
        config = {
            "config": args.config_path,
            "ckpt": args.ckpt_path,
            "use_hypernetwork": use_hyper,
            "clip_size": args.clip_size,
            "lora_rank": args.lora_rank,
            "lora_alpha": args.lora_alpha,
            "target_modules": args.target_modules,
            "iterations": args.iterations,
            "lr": args.lr,
            "image_size": args.image_size,
            "ddim_steps": args.ddim_steps,
            "ddim_eta": args.ddim_eta,
            "start_guidance": args.start_guidance,
            "negative_guidance": args.negative_guidance,
            "final_loss": losses[-1] if losses else None,
            "average_loss": (sum(losses) / len(losses)) if losses else None,
        }
        with open(os.path.join(run_dir, "train_config.json"), "w") as f:
            json.dump(config, f, indent=2)

    if accelerator.is_local_main_process and args.use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
