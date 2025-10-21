#!/usr/bin/env python3

# Standard library
import json
import os
import random
from functools import partial
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

# Third-party core
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

# PyTorch ecosystem
from torch.utils.data import DataLoader
from torchvision.transforms.functional import to_pil_image

# Accelerate
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed as hf_set_seed

# Transformers
from transformers import CLIPTextModel, CLIPTokenizer

# Local modules
from config import TrainingConfig, parse_args
from data_utils import TargetReferenceDataset, collate_prompts
from hyper_lora import HyperLora, HyperLoRALinear, inject_hyper_lora, inject_hyper_lora_nsfw
from ldm.models.diffusion.ddimcopy import DDIMSampler
from ldm.util import instantiate_from_config
from sampling import sample_model
from utils import get_models, print_trainable_parameters  # DO NOT import set_seed here to avoid clashes
from wandb_logger import WandbLogger


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



def main():
    # Parse arguments and create configuration
    args = parse_args()
    config = TrainingConfig.from_args(args)
    
    # Validate configuration
    config.validate()

    # Load neutral concepts from JSON config
    with open(config.neutral_concepts_file, 'r') as f:
        concepts_config = json.load(f)
        neutral_concepts = concepts_config.get("neutral_concepts", [])

    if not neutral_concepts:
        print(f"Warning: No neutral concepts found in {config.neutral_concepts_file}")

    # Print basic config
    print("=== LoRA/HyperLoRA Fine-tuning (Accelerate) ===")
    print(f"Config: {config.config_path}")
    print(f"Checkpoint: {config.ckpt_path}")
    print(f"Output dir: {config.output_dir}")
    print(f"Iterations: {config.iterations}  |  LR: {config.lr}  |  Accum: {config.gradient_accumulation_steps}")
    print(f"Image size: {config.image_size}  |  DDIM steps: {config.ddim_steps}  |  eta: {config.ddim_eta}")
    print("=" * 48)

    # Seed
    if config.seed is not None:
        hf_set_seed(config.seed)

    # Accelerate project config
    accelerator_project_config = ProjectConfiguration(
        project_dir=config.output_dir,
        logging_dir=config.logging_dir,
    )

    accelerator = Accelerator(
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        mixed_precision=config.mixed_precision,  # None -> use accelerate config
        log_with="wandb" if config.use_wandb else None,
        project_config=accelerator_project_config,
    )

    config.lr = (
            config.lr
            * config.gradient_accumulation_steps
            * config.batch_size
            * accelerator.num_processes
    )

    #logger = get_logger(__name__)
    is_main = accelerator.is_main_process

    # Initialize W&B logger
    wandb_logger = WandbLogger(enabled=(is_main and config.use_wandb))
    wandb_logger.init_tracker(config=config.to_dict(), project_name="UnGuide")

    # Data
    data_dir = config.data_dir
    ds = TargetReferenceDataset(data_dir)
    ds_loader = DataLoader(ds, batch_size=1, shuffle=True, collate_fn=collate_prompts)

    # Models (original + trainable clone)
    if is_main:
        os.makedirs(os.path.join(config.output_dir, "tmp"), exist_ok=True)

    model_orig, sampler_orig, model, sampler_unused = get_models(
        config.config_path, config.ckpt_path, accelerator.device
    )

    # Freeze original model
    for p in model_orig.model.diffusion_model.parameters():
        p.requires_grad = False
    model_orig.eval()

    for p in model.model.diffusion_model.parameters():
        p.requires_grad = False

    # Add attribute used downstream
    model.current_conditioning = None

    # Inject HyperLoRA/LoRA BEFORE prepare(), then build optimizer on trainable params
    use_hyper = True  # your script forces hypernetwork on; keep same behavior
    hyper_lora_factory = partial(
        HyperLoRALinear,
        clip_size=config.clip_size,
        rank=config.lora_rank,
        alpha=config.lora_alpha,
    )
    hyper_lora_layers = inject_hyper_lora(
        model.model.diffusion_model, config.target_modules, hyper_lora_factory
    )
    for layer in hyper_lora_layers:
        layer.set_parent_model(model)


    # Optimizer on trainable (LoRA) params only
    trainable_params = list(filter(lambda p: p.requires_grad, model.model.diffusion_model.parameters()))
    if is_main:
        print(f"Total trainable parameter tensors: {len(trainable_params)}")
        print_trainable_parameters(model)

    optimizer = torch.optim.Adam(trainable_params, lr=config.lr)

    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[300], gamma=0.5
    )

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
    quick_sampler = create_quick_sampler(base, sampler, config.image_size, config.ddim_steps, config.ddim_eta)

    sampler_orig = DDIMSampler(model_orig)

    # Optionally log a baseline image (main only)
    if is_main:
        imgs0 = generate_and_save_sd_images(
            model=model_orig,
            sampler=sampler_orig,
            prompt=ds[0]["target"],
            device=accelerator.device,
            steps=50,
            out_dir=os.path.join(config.output_dir, "tmp"),
            prefix="orig_",
        )
        if imgs0 is not None:
            im0 = (imgs0[0].clamp(0, 1) * 255).round().to(torch.uint8).cpu()
            wandb_logger.log_image(im0, caption="baseline", step=0, key="baseline")
    # Training
    criterion = torch.nn.MSELoss()
    losses = []

    pbar = tqdm(range(config.iterations), disable=not accelerator.is_local_main_process)
    for i in pbar:
        for sample_ids, sample in enumerate(ds_loader):
            # Get conditional embeddings (strings) directly for LDM
            emb_0 = base.get_learned_conditioning(sample["reference"])
            emb_p = base.get_learned_conditioning(sample["target"])
            emb_n = base.get_learned_conditioning(sample["target"])

            optimizer.zero_grad(set_to_none=True)

            # random timestep mapping (keep your logic)
            t_enc = torch.randint(config.ddim_steps, (1,), device=accelerator.device)
            og_num = round((int(t_enc) / config.ddim_steps) * 1000)
            og_num_lim = round((int(t_enc + 1) / config.ddim_steps) * 1000)
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

            inputs = encode(sample["target"])
            #print('!!! ', sample["target"])
            inputs_other = encode("a photo of the car")
            inputs_other2 = encode("a photo of the castle")
            inputs_airplane = encode("a photo of the airplane")
            with torch.no_grad():
                cond_target = clip_text_encoder(inputs).pooler_output.detach()
                cond_other = clip_text_encoder(inputs_other).pooler_output.detach()
                cond_other2 = clip_text_encoder(inputs_other2).pooler_output.detach()
                cond_airplane = clip_text_encoder(inputs_airplane).pooler_output.detach()
                #cond_ref    = clip_text_encoder(inputs[1]).pooler_output.detach()

            # pass both to model for HyperLoRA
            base = accelerator.unwrap_model(model)  # the actual Module used in forward
            base.current_conditioning = cond_target
            base.time_step = int(torch.randint(0, 149, (1,), device=accelerator.device))
            # starting latent code
            start_code = torch.randn(
                (1, 4, config.image_size // 8, config.image_size // 8),
                device=accelerator.device,
            )
            with accelerator.accumulate(model):
                if 'neutral.json' in sample['file']:
                    base.time_step = 0
                    neutral_category = random.choice(neutral_concepts)
                    neutral_prompt = f"A photo of the {neutral_category}"
                    inputs_neutral = encode(neutral_prompt)
                    with torch.no_grad():
                        base.current_conditioning = clip_text_encoder(inputs_neutral).pooler_output.detach()
                    #base.current_conditioning = (1- alpha) * cond_target + alpha * cond_cat

                    z = quick_sampler(emb_p, config.start_guidance, start_code, int(t_enc))
                    emb_cat = base.get_learned_conditioning("A photo of the airplane")
                    _ = accelerator.unwrap_model(model).apply_model(z, t_enc_ddpm, emb_cat)
                    tensors_flat_t_live = flatten_live_tensors(model, accelerator)
                    #with torch.no_grad():
                    #    l2 = tensors_flat_t_live.float().norm(p=2).item()  # L2 norm
                    #accelerator.print(f"||tensors_flat_t_live||_2 = {l2:.6f}")

                    base.time_step = 150
                    _ = base.apply_model(z, t_enc_ddpm, emb_cat)
                    tensors_flat_t1_live = flatten_live_tensors(model, accelerator)
                    delta_live = tensors_flat_t1_live - tensors_flat_t_live
                    loss = (delta_live ** 2).mean()
                    loss_for_backward = loss / accelerator.gradient_accumulation_steps
                    #print('!!!! ', loss_for_backward)
                else:
                    with torch.no_grad():
                        z = quick_sampler(emb_p, config.start_guidance, start_code, int(t_enc))
                        e_0 = model_orig.apply_model(z, t_enc_ddpm, emb_0)  # reference (stopgrad)
                        e_p = model_orig.apply_model(z, t_enc_ddpm, emb_p)  # target   (stopgrad)

                    # prediction for trainable model (needs grads)
                    e_n = accelerator.unwrap_model(model).apply_model(z, t_enc_ddpm, emb_n)

                    # targets and loss
                    e_0.requires_grad_(False)
                    e_p.requires_grad_(False)
                    target = e_0 - (config.negative_guidance * (e_p - e_0))
                    loss = criterion(e_n, target)  # per-rank scalar tensor

                    # ---- BACKWARD (per micro-step) ----
                    # Scale the loss for gradient accumulation so the effective grad equals the true average
                    loss_for_backward = loss / accelerator.gradient_accumulation_steps
                    accelerator.backward(loss_for_backward, retain_graph=True)
                    base = accelerator.unwrap_model(model)
                    dev = next(base.parameters()).device

                    recs = collect_hyperlora_tensors_and_grads(model, accelerator)
                    pack = concat_grads_and_tensors(recs, device=dev)  # has 'grads_flat', 'tensors_flat', etc.

                    # Target step: Δθ ≈ -lr * g_t  (keep target detached)
                    grads_flat_t = -1 * config.internal_lr * pack["grads_flat"].detach()

                    # --- LIVE anchor at t (no detach!) ---
                    tensors_flat_t_live = flatten_live_tensors(model, accelerator)
                    #print('!! ', pack["grads_flat"].shape, pack["tensors_flat"].shape, tensors_flat_t_live.shape)

                    # Clear grads before the next forward
                    #optimizer.zero_grad(set_to_none=True)
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
                    if accelerator.is_local_main_process:
                        assert len(lrs_after) == 1
                        wandb_logger.log_metrics(
                            {"lr_group": lrs_after[0]},
                            step=i
                        )
            # Optional image logging
            if (
                accelerator.is_local_main_process
                and i >= config.log_from
                and i % 10 == 0
                and sample_ids == 0
            ):
                base.time_step = 150
                base.current_conditioning = cond_airplane
                imgs = generate_and_save_sd_images(
                    model=base,
                    sampler=sampler,
                    prompt="a photo of the airplane",
                    device=accelerator.device,
                    steps=50,
                    out_dir=os.path.join(config.output_dir, "tmp"),
                    prefix=f"unl_{i}_",
                )
                if imgs is not None:
                    caption = f"target: airplane"
                    im0 = (imgs[0].clamp(0, 1) * 255).round().to(torch.uint8).cpu()
                    wandb_logger.log_image(im0, caption=caption, step=i, key="sample")
                base.current_conditioning = cond_other
                imgs = generate_and_save_sd_images(
                    model=base,
                    sampler=sampler,
                    prompt="a photo of the car",
                    device=accelerator.device,
                    steps=50,
                    out_dir=os.path.join(config.output_dir, "tmp"),
                    prefix=f"unl_{i}_",
                )
                if imgs is not None:
                    caption = f"target: car"
                    im0 = (imgs[0].clamp(0, 1) * 255).round().to(torch.uint8).cpu()
                    wandb_logger.log_image(im0, caption=caption, step=i, key="sample (other)")
                base.current_conditioning = cond_other2
                #base.time_step = 0
                #print('---!!! ', base.time_step, accelerator.unwrap_model(model).time_step)
                imgs = generate_and_save_sd_images(
                    model=base,
                    sampler=sampler,
                    prompt="a photo of the car",
                    device=accelerator.device,
                    steps=50,
                    out_dir=os.path.join(config.output_dir, "tmp"),
                    prefix=f"unl_{i}_",
                )
                if imgs is not None:
                    caption = f"target: car"
                    im0 = (imgs[0].clamp(0, 1) * 255).round().to(torch.uint8).cpu()
                    wandb_logger.log_image(im0, caption=caption, step=i, key="sample (other) from castle")

            with torch.no_grad():
                loss_reduced = accelerator.gather(loss.detach()).mean()

            loss_value = float(loss_reduced.item())
            losses.append(loss_value)

            if accelerator.is_local_main_process:
                wandb_logger.log_metrics({"loss": loss_value}, step=i)

            if accelerator.is_local_main_process:
                pbar.set_postfix({"loss": f"{loss_value:.6f}"})

        # Save LoRA/HyperLoRA weights each iteration (or move outside loop if you prefer)
        accelerator.wait_for_everyone()
        if accelerator.is_local_main_process:
            save_dir = os.path.join(config.output_dir, f"rank_{config.lora_rank}_it_{config.iterations}_lr_{config.lr}_sg_{config.start_guidance}_ng_{config.negative_guidance}_ddim_{config.ddim_steps}_" + ("hyper" if use_hyper else "lora"))
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
        run_dir = os.path.dirname(lora_path) if losses else config.output_dir
        config_dict = {
            "config": config.config_path,
            "ckpt": config.ckpt_path,
            "use_hypernetwork": use_hyper,
            "clip_size": config.clip_size,
            "lora_rank": config.lora_rank,
            "lora_alpha": config.lora_alpha,
            "target_modules": config.target_modules,
            "iterations": config.iterations,
            "lr": config.lr,
            "image_size": config.image_size,
            "ddim_steps": config.ddim_steps,
            "ddim_eta": config.ddim_eta,
            "start_guidance": config.start_guidance,
            "negative_guidance": config.negative_guidance,
            "final_loss": losses[-1] if losses else None,
            "average_loss": (sum(losses) / len(losses)) if losses else None,
        }
        with open(os.path.join(run_dir, "train_config.json"), "w") as f:
            json.dump(config_dict, f, indent=2)

    # Cleanup W&B
    wandb_logger.finish()


if __name__ == "__main__":
    main()
