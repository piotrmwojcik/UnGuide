#!/usr/bin/env python3
import argparse
import json
import os
from functools import partial
from pathlib import Path

import matplotlib.pyplot as plt  # (unused, but kept if you uncomment stats)
import numpy as np
import torch
import wandb
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
from torchvision.transforms.functional import to_pil_image
from transformers import CLIPTextModel, CLIPTokenizer
from tqdm import tqdm

from data_utils import TargetReferenceDataset, collate_prompts
from ldm.models.diffusion.ddimcopy import DDIMSampler
from ldm.util import instantiate_from_config  # (kept if your get_models uses it)
from hyper_lora import (HyperLoRALinear, inject_hyper_lora,
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
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size (per device) for the training dataloader.",
    )


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
    Concatenate CURRENT tensors from all HyperLora layers WITHOUT detaching.
    This preserves autograd so loss can backprop through them.
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


def _iter_hyperlora_layers(root: nn.Module) -> Iterator[Tuple[str, Any]]:
    for name, m in root.named_modules():
        if isinstance(m, HyperLoRALinear):
            yield name + ".hyper_lora", m.hyper_lora
        elif isinstance(m, HyperLora):
            yield name, m


from typing import Iterator, Tuple, Dict, Any

from typing import Dict, Any, Iterator, Tuple
import torch
import torch.nn as nn
from typing import Dict, Any, List, Tuple

# assumes you have these classes
from hyper_lora import HyperLora, HyperLoRALinear


def _iter_hyperlora_layers(root: nn.Module) -> Iterator[Tuple[str, HyperLora]]:
    """Yield (qualified_name, HyperLora) whether wrapped or direct."""
    for name, m in root.named_modules():
        if isinstance(m, HyperLoRALinear):
            yield name + ".hyper_lora", m.hyper_lora
        elif isinstance(m, HyperLora):
            yield name, m

def collect_hyperlora_tensors_and_grads(
    model_wrapped: nn.Module, accelerator, verbose: bool = True, all_ranks: bool = True
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

    #logger = get_logger(__name__)
    is_main = accelerator.is_main_process

    # Trackers (W&B/TB/etc.) — initialize after Accelerator so it attaches run metadata
    if is_main and args.use_wandb and ("wandb" in str(args.report_to) or args.report_to == "all"):
        wandb.init(project="UnGuide", name="training", config=vars(args))

    # Data
    data_dir = args.data_dir
    ds = TargetReferenceDataset(data_dir)
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

    # Add attribute used downstream
    model.current_conditioning = None

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
    for layer in hyper_lora_layers:
        layer.set_parent_model(model)


    # Optimizer on trainable (LoRA) params only
    trainable_params = list(filter(lambda p: p.requires_grad, model.model.diffusion_model.parameters()))
    if is_main:
        print(f"Total trainable parameter tensors: {len(trainable_params)}")
        print_trainable_parameters(model)

    optimizer = torch.optim.Adam(trainable_params, lr=args.lr)

    # Prepare for DDP / Mixed precision
    model, optimizer, ds_loader = accelerator.prepare(model, optimizer, ds_loader)

    base = accelerator.unwrap_model(model)
    for layer in hyper_lora_layers:
        layer.set_parent_model(base)

        # Create sampler AFTER prepare so it uses the wrapped model
    sampler = DDIMSampler(base)

    # Tokenizer + CLIP text encoder (inference-only; keep unwrapped)
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
    clip_text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14").to(accelerator.device).eval()

    # Quick sampler
    quick_sampler = create_quick_sampler(base, sampler, args.image_size, args.ddim_steps, args.ddim_eta)

    sampler_orig = DDIMSampler(model_orig)

    # Optionally log a baseline image (main only)
    if is_main:
        imgs0 = generate_and_save_sd_images(
            model=model_orig,
            sampler=sampler_orig,
            prompt=ds[0]["target"],
            device=accelerator.device,
            steps=50,
            out_dir=os.path.join(args.output_dir, "tmp"),
            prefix="orig_",
        )
        if args.use_wandb and imgs0 is not None:
            im0 = (imgs0[0].clamp(0, 1) * 255).round().to(torch.uint8).cpu()
            wandb.log({"baseline": wandb.Image(to_pil_image(im0))}, step=0)
    # Training
    criterion = torch.nn.MSELoss()
    losses = []

    pbar = tqdm(range(args.iterations), disable=not accelerator.is_local_main_process)
    for i in pbar:
        for sample_ids, sample in enumerate(ds_loader):
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

            inputs = (encode(sample["target"]), encode(sample["reference"]))
            with torch.no_grad():
                cond_target = clip_text_encoder(inputs[0]).pooler_output.detach()
                cond_ref    = clip_text_encoder(inputs[1]).pooler_output.detach()

            # pass both to model for HyperLoRA
            base = accelerator.unwrap_model(model)  # the actual Module used in forward
            base.current_conditioning = (cond_target, cond_ref)
            base.time_step = int(torch.randint(0, 499, (1,), device=accelerator.device))
            # starting latent code
            start_code = torch.randn(
                (1, 4, args.image_size // 8, args.image_size // 8),
                device=accelerator.device
            )
            with accelerator.accumulate(model):  # pass the WRAPPED model here
                with torch.no_grad():
                    z = quick_sampler(emb_p, args.start_guidance, start_code, int(t_enc))
                    e_0 = model_orig.apply_model(z, t_enc_ddpm, emb_0)  # reference (stopgrad)
                    e_p = model_orig.apply_model(z, t_enc_ddpm, emb_p)  # target   (stopgrad)

                # prediction for trainable model (needs grads)
                e_n = accelerator.unwrap_model(model).apply_model(z, t_enc_ddpm, emb_n)

                # targets and loss
                e_0.requires_grad_(False)
                e_p.requires_grad_(False)
                target = e_0 - (args.negative_guidance * (e_p - e_0))
                loss = criterion(e_n, target)  # per-rank scalar tensor

                # ---- BACKWARD (per micro-step) ----
                # Scale the loss for gradient accumulation so the effective grad equals the true average
                loss_for_backward = loss / accelerator.gradient_accumulation_steps
                accelerator.backward(loss_for_backward)
                base = accelerator.unwrap_model(model)
                dev = next(base.parameters()).device

                recs = collect_hyperlora_tensors_and_grads(model, accelerator)
                pack = concat_grads_and_tensors(recs, device=dev)  # has 'grads_flat', 'tensors_flat', etc.

                # Target step: Δθ ≈ -lr * g_t  (keep target detached)
                grads_flat_t = (-args.lr) * pack["grads_flat"].detach()

                # --- LIVE anchor at t (no detach!) ---
                tensors_flat_t_live = flatten_live_tensors(model, accelerator)
                print('!! ', pack["grads_flat"].shape, pack["tensors_flat"].shape, tensors_flat_t_live.shape)

                # Clear grads before the next forward
                optimizer.zero_grad(set_to_none=True)
                for _, hl in _iter_hyperlora_layers(base):
                    xL = getattr(hl, "_last_x_L", None)
                    xR = getattr(hl, "_last_x_R", None)
                    if xL is not None: xL.grad = None
                    if xR is not None: xR.grad = None

                # Advance time and run forward to populate new live tensors
                base.time_step = base.time_step + 1
                _ = base.apply_model(z, t_enc_ddpm, emb_n)

                # LIVE vector at t+1 (keeps graph)
                tensors_flat_t1_live = flatten_live_tensors(model, accelerator)

                # Match the SGD step: (θ_{t+1} - θ_t) ≈ -lr * g_t
                delta_live = tensors_flat_t1_live - tensors_flat_t_live

                # e.g., MSE to the target step
                loss = criterion(delta_live, grads_flat_t)
                loss_for_backward = loss / accelerator.gradient_accumulation_steps
                accelerator.backward(loss_for_backward)
                # def _fmt_tensor(t):
                #     if t is None:
                #         return "None"
                #     if t.numel() == 1:
                #         return f"val={t.item():.4e}"
                #     # avoid NaNs if size==1 along axes
                #     try:
                #         mean = t.mean().item()
                #         std = t.std(unbiased=False).item()
                #     except Exception:
                #         mean, std = float("nan"), float("nan")
                #     return f"shape={tuple(t.shape)} | ||·||={t.norm().item():.4e} | mean={mean:.4e} | std={std:.4e}"
                #
                # # tiny diagnostic print (norms)
                # for name, g in grads.items():
                #     a = g.get("alpha")
                #     a_g = g.get("alpha_grad")
                #     xL = g.get("x_L")
                #     xL_g = g.get("x_L_grad")
                #     xR = g.get("x_R")
                #     xR_g = g.get("x_R_grad")
                #     WL = g.get("left_head_w")
                #     WL_g = g.get("left_head_w_grad")
                #     WR = g.get("right_head_w")
                #     WR_g = g.get("right_head_w_grad")
                #     bL = g.get("left_head_b")
                #     bL_g = g.get("left_head_b_grad")
                #     bR = g.get("right_head_b")
                #     bR_g = g.get("right_head_b_grad")
                #
                #     print(f"[{name}] ------------------------------", flush=True)
                #     print(f"  alpha    : {_fmt_tensor(a)}", flush=True)
                #     print(f"  d(alpha) : {_fmt_tensor(a_g)}", flush=True)
                #     print(f"  x_L      : {_fmt_tensor(xL)}", flush=True)
                #     print(f"  d(x_L)   : {_fmt_tensor(xL_g)}", flush=True)
                #     print(f"  x_R      : {_fmt_tensor(xR)}", flush=True)
                #     print(f"  d(x_R)   : {_fmt_tensor(xR_g)}", flush=True)
                #     print(f"  W_L      : {_fmt_tensor(WL)}", flush=True)
                #     print(f"  d(W_L)   : {_fmt_tensor(WL_g)}", flush=True)
                #     print(f"  b_L      : {_fmt_tensor(bL)}", flush=True)
                #     print(f"  d(b_L)   : {_fmt_tensor(bL_g)}", flush=True)
                #     print(f"  W_R      : {_fmt_tensor(WR)}", flush=True)
                #     print(f"  d(W_R)   : {_fmt_tensor(WR_g)}", flush=True)
                #     print(f"  b_R      : {_fmt_tensor(bR)}", flush=True)
                #     print(f"  d(b_R)   : {_fmt_tensor(bR_g)}", flush=True)

                # ---- OPTIMIZER STEP (only on last micro-step) ----
                if accelerator.sync_gradients:
                    # (optional) gradient clipping
                    # accelerator.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
            # Optional image logging
            if (
                is_main
                and args.use_wandb
                and i >= args.log_from
                and i % 10 == 0
                and sample_ids == 0
            ):
                base.time_step = 150
                imgs = generate_and_save_sd_images(
                    model=base,
                    sampler=sampler,
                    prompt=sample["target"][0],
                    device=accelerator.device,
                    steps=50,
                    out_dir=os.path.join(args.output_dir, "tmp"),
                    prefix=f"unl_{i}_",
                )
                if imgs is not None:
                    caption = f"target: {sample['target'][0]}"
                    im0 = (imgs[0].clamp(0, 1) * 255).round().to(torch.uint8).cpu()
                    wandb.log({"sample": wandb.Image(to_pil_image(im0), caption=caption)}, step=i)

            with torch.no_grad():
                loss_reduced = accelerator.gather(loss.detach()).mean()

            loss_value = float(loss_reduced.item())
            losses.append(loss_value)

            if is_main and args.use_wandb:
                wandb.log({"loss": loss_value, "iter": i}, step=i)

            if accelerator.is_local_main_process:
                pbar.set_postfix({"loss": f"{loss_value:.6f}"})

        # Save LoRA/HyperLoRA weights each iteration (or move outside loop if you prefer)
        accelerator.wait_for_everyone()
        if is_main:
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
    if is_main:
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

    if is_main and args.use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
