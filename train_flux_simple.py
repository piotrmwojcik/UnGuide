#!/usr/bin/env python3
"""
Simplified training script that reads all configuration from a YAML file.
Usage: python train_simple.py --config configs/train_config_example.yaml

All training parameters (model paths, hyperparameters, concepts, etc.) are specified in the YAML config.
No additional command-line arguments are needed.
"""

import argparse
import json
import os
import random
from typing import Any, Callable, Dict, List, Optional, Union
from torchvision.transforms.functional import to_pil_image
import copy
from pathlib import Path
from functools import partial

import pandas as pd
import torch
import torch.nn as nn
import wandb
import yaml
from diffusers import FluxPipeline
from tools.prompt_process import encode_prompt
from tools.scheduler_process import FlowMatchEulerDiscreteScheduler
from torchvision.transforms.functional import to_tensor
from accelerate import Accelerator
from tools.scheduler_process import FlowMatchEulerDiscreteScheduler
from utils_flux.esd_utils import latent_sample, predict_noise, flux_pack_latents, _prepare_latent_image_ids
from transformers import CLIPTokenizer, PretrainedConfig, T5TokenizerFast
from accelerate.utils import ProjectConfiguration, set_seed as hf_set_seed
from torchvision.transforms.functional import to_pil_image
from transformers import CLIPTextModel, CLIPTokenizer
from tqdm import tqdm

from diffusers import FluxPipeline
from diffusers import (
    AutoencoderKL,
    FlowMatchEulerDiscreteScheduler,
    FluxPipeline,
    FluxTransformer2DModel,
)

from hyper_lora import HyperLoRALinear, HypernetworkManager, inject_hyper_lora
from ldm.models.diffusion.ddimcopy import DDIMSampler
from sampling import sample_model
from utils import print_trainable_parameters


class CombinedCFGModel:
    """Wrapper that uses different models for conditional and unconditional passes."""

    def __init__(self, cond_model, uncond_model):
        self.cond_model = cond_model
        self.uncond_model = uncond_model
        self.device = cond_model.device

    def apply_model(self, x, t, c):
        # When DDIMSampler uses guidance, it concatenates [uncond, cond] inputs
        # We split and route to different models
        b2 = x.shape[0]
        assert b2 % 2 == 0
        b = b2 // 2

        x_uncond = x[:b]
        x_cond = x[b:]
        t_uncond = t[:b]
        t_cond = t[b:]

        # Split conditioning
        if isinstance(c, dict):
            c_uncond = {}
            c_cond = {}
            for k in c:
                if isinstance(c[k], list):
                    c_uncond[k] = [v[:b] for v in c[k]]
                    c_cond[k] = [v[b:] for v in c[k]]
                else:
                    c_uncond[k] = c[k][:b]
                    c_cond[k] = c[k][b:]
        else:
            c_uncond = c[:b]
            c_cond = c[b:]

        # Route unconditional to model_orig, conditional to model (with LoRA)
        out_uncond = self.uncond_model.apply_model(x_uncond, t_uncond, c_uncond)
        out_cond = self.cond_model.apply_model(x_cond, t_cond, c_cond)

        return torch.cat([out_uncond, out_cond], dim=0)

    def get_learned_conditioning(self, prompts):
        return self.cond_model.get_learned_conditioning(prompts)

    def decode_first_stage(self, z):
        return self.cond_model.decode_first_stage(z)

    def eval(self):
        self.cond_model.eval()
        self.uncond_model.eval()
        return self

    def __getattr__(self, name):
        return getattr(self.cond_model, name)


def prompt_augmentation(content, augment=True):
    """Generate augmented prompts for a given concept."""
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
    else:
        return [content]


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Extract the first config key (e.g., 'MACE')
    config_name = list(config.keys())[0]
    return config[config_name], config_name


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Simplified HyperLoRA Training for Stable Diffusion"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML configuration file",
    )
    return parser.parse_args()


def create_quick_sampler(model, sampler, image_size: int, ddim_steps: int, ddim_eta: float):
    """Create a quick sampling function with fixed parameters."""
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


def load_text_encoders(class_one, class_two, pretrained_model_name_or_path):
    text_encoder_one = class_one.from_pretrained(
        pretrained_model_name_or_path, subfolder="text_encoder", revision=None, variant=None
    )
    text_encoder_two = class_two.from_pretrained(
        pretrained_model_name_or_path, subfolder="text_encoder_2", revision=None, variant=None
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


def compute_text_embeddings(prompts, text_encoders, tokenizers, device, max_sequence_length=256):
    # prompts: List[str] or str
    if isinstance(prompts, str):
        prompts = [prompts]

    prompt_embeds, pooled_prompt_embeds, text_ids = encode_prompt(
        text_encoders, tokenizers, prompts, max_sequence_length
    )
    return (
        prompt_embeds.to(device),
        pooled_prompt_embeds.to(device),
        text_ids.to(device),
    )


def generate_images(
        sampler,
        model,
        prompt: str,
        device: torch.device,
        steps: int = 50,
        eta: float = 0.0,
        batch_size: int = 1,
        start_code: torch.Tensor = None,
        guidance_scale: float = 7.5,
):
    """
    Generate images with CFG from a CompVis SD model + DDIMSampler.
    Uses the same approach as generate_images_nsfw_cfg.py.
    """
    if start_code is None:
        start_code = torch.randn(batch_size, 4, 64, 64, device=device)

    model.eval()
    with torch.no_grad(), torch.autocast(device_type=device.type, enabled=(device.type == "cuda")):
        cond = model.get_learned_conditioning([prompt] * start_code.shape[0])
        uncond = model.get_learned_conditioning([""] * start_code.shape[0])

        samples, _ = sampler.sample(
            S=steps,
            conditioning={"c_crossattn": [cond]},
            batch_size=start_code.shape[0],
            shape=start_code.shape[1:],
            verbose=False,
            unconditional_guidance_scale=guidance_scale,
            unconditional_conditioning={"c_crossattn": [uncond]},
            eta=eta,
            x_T=start_code,
        )
        decoded = model.decode_first_stage(samples)
        decoded = (decoded + 1.0) / 2.0
        decoded = torch.clamp(decoded, 0.0, 1.0)
        return decoded


def main():
    args = parse_args()

    # Load configuration
    config, config_name = load_config(args.config)
    print(f"=== Training with config: {config_name} ===")
    print(f"Config file: {args.config}")

    # Extract key parameters with defaults
    learning_rate = config.get('learning_rate', 1e-5)
    max_train_steps = config.get('max_train_steps', 120)
    hyper_train_steps = config.get('hyper_train_steps', 500)  # Steps for hypernetwork context
    rank = config.get('rank', 1)
    lora_alpha = config.get('lora_alpha', 8)  # LoRA alpha parameter
    seed = config.get('seed', 2024)
    resolution = config.get('resolution', 512)
    use_pooler = config.get('use_pooler', True)
    use_orig_concat = config.get('use_orig_concat', False)
    gradient_accumulation_steps = config.get('gradient_accumulation_steps', 1)

    # Multi-concept configuration
    concepts = config.get('concepts', [])
    mapping_concept = config.get('mapping_concept', [])
    retain_csv_path = config.get('retain_csv_path', None)  # Path to CSV with retain prompts

    # Augmentation flags
    augment_target = config.get('augment_target', True)  # Whether to augment target concepts
    augment_retain = config.get('augment_retain', False)  # Whether to augment retain prompts from CSV

    # Paths
    output_dir = config.get('output_dir', './output')
    final_save_path = config.get('final_save_path', './saved_model/LoRA_fusion_model')
    pretrained_model_name_or_path = config.get('pretrained_model_name_or_path', "black-forest-labs/FLUX.1-dev")

    # Training settings
    ddim_steps = 50
    ddim_eta = 0.0
    negative_guidance = config.get('negative_guidance', 2.0)
    guidance_scale = config.get('guidance_scale', 7.5)
    start_guidance = config.get('guidance_scale', 9.0)
    internal_lr = config.get('internal_lr', 1e-4)  # Simulated lr for hypernetwork gradient matching

    # Diagnostic prompts for image generation during training
    diagnostic_prompts = config.get('diagnostic_prompts', [])
    if not diagnostic_prompts:
        # Default diagnostic prompts if none provided
        diagnostic_prompts = [
            f"a photo of {concepts[0]}" if concepts else "a photo of a person",
            "a photo of a cat",
            "a photo of a car"
        ]

    print(f"Training steps: {max_train_steps}")
    print(f"Hypernetwork steps: {hyper_train_steps}")
    print(f"Learning rate: {learning_rate}")
    print(f"LoRA rank: {rank}")
    print(f"LoRA alpha: {lora_alpha}")
    print(f"Target concepts: {len(concepts)}")
    print("=" * 48)

    # Set seed
    if seed is not None:
        hf_set_seed(seed)

    # Setup Accelerator
    accelerator_project_config = ProjectConfiguration(
        project_dir=output_dir,
        logging_dir=config.get('logging_dir', 'logs'),
    )

    accelerator = Accelerator(
        gradient_accumulation_steps=gradient_accumulation_steps,
        mixed_precision=config.get('mixed_precision', None),
        log_with=config.get('report_to', 'wandb'),
        project_config=accelerator_project_config,
    )

    is_main = accelerator.is_main_process

    # Initialize W&B if needed
    use_wandb = config.get('report_to') == 'wandb'
    if is_main and use_wandb:
        wandb.init(
            project="UnGuide",
            name=f"{config_name}_training",
            config=config
        )

    tokenizer_one = CLIPTokenizer.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="tokenizer",
        revision=None,
    )
    tokenizer_two = T5TokenizerFast.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="tokenizer_2",
        revision=None,
    )

    # import correct text encoder classes
    text_encoder_cls_one = import_model_class_from_model_name_or_path(
        pretrained_model_name_or_path, None
    )
    text_encoder_cls_two = import_model_class_from_model_name_or_path(
        pretrained_model_name_or_path, None, subfolder="text_encoder_2"
    )

    noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        pretrained_model_name_or_path, subfolder="scheduler"
    )
    text_encoder_one, text_encoder_two = load_text_encoders(text_encoder_cls_one, text_encoder_cls_two,
                                                            pretrained_model_name_or_path)
    vae = AutoencoderKL.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="vae",
        revision=None,
        variant=None,
    )

    weight_dtype = torch.bfloat16 if accelerator.device.type == "cuda" else torch.float32

    model = FluxTransformer2DModel.from_pretrained(
        pretrained_model_name_or_path, torch_dtype=weight_dtype,
        subfolder="transformer", revision=None, variant=None
    ).to(accelerator.device)

    model.requires_grad_(False)
    vae.requires_grad_(False)
    text_encoder_one.requires_grad_(False)
    text_encoder_two.requires_grad_(False)

    tokenizers = [tokenizer_one, tokenizer_two]
    text_encoders = [text_encoder_one, text_encoder_two]

    # # Load models
    # model_orig, sampler_orig, model, sampler_unused = get_models(
    #     model_config_path, pretrained_model_path, accelerator.device
    # )
    #
    # # Freeze original model
    # for p in model_orig.model.diffusion_model.parameters():
    #     p.requires_grad = False
    # model_orig.eval()
    #
    # # Freeze trainable model backbone
    # for p in model.model.diffusion_model.parameters():
    #     p.requires_grad = False
    #
    # # Setup HyperLoRA
    model.hyper = HypernetworkManager()

    clip_size = 768 if use_pooler else 512
    target_modules = ["attn.add_k_proj", "attn.add_q_proj"]

    hyper_lora_factory = partial(
        HyperLoRALinear,
        clip_size=clip_size,
        rank=rank,
        alpha=lora_alpha,
        train_steps=hyper_train_steps,
        use_orig_concat=use_orig_concat,
        dtype=torch.bfloat16,
    )

    hyper_lora_layers = inject_hyper_lora(
        model, target_modules, hyper_lora_factory
    )

    for layer_name, layer in hyper_lora_layers:
        layer.set_parent_model(model)
        layer.to(dtype=torch.bfloat16)  # converts parameters + buffers inside the injected module

    # Setup optimizer
    trainable_params = [p for p in model.parameters() if p.requires_grad]

    if is_main:
        print(f"Total trainable parameter tensors: {len(trainable_params)}")
        print_trainable_parameters(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    gamma = config.get('gamma', 0.9)  # Weight for removal loss
    step_size = config.get('step_size', 300)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[step_size], gamma=gamma
    )

    # Prepare for distributed training
    model, optimizer = accelerator.prepare(model, optimizer)

    # Register HyperLoRA layers after prepare
    for layer_name, layer in hyper_lora_layers:
        layer.set_parent_model(accelerator.unwrap_model(model))
        accelerator.unwrap_model(model).hyper.add_hyperlora(layer_name, layer.hyper_lora)

    # Setup CLIP for conditioning
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
    clip_text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14").to(accelerator.device).eval()

    def encode(text: str):
        return tokenizer(
            text,
            max_length=tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        ).to(accelerator.device).input_ids

    # Prepare concept embeddings
    target_concepts = []
    # concepts is now a simple list of concept strings
    for concept_text in concepts:
        target_concepts.append(concept_text)

    print(f"Target concepts for removal: {target_concepts}")
    print(f"Total target concepts: {len(target_concepts)}")

    # Create concept embeddings
    target_embeddings = []
    for concept in target_concepts:
        inputs = encode(concept)
        with torch.no_grad():
            if use_pooler:
                emb = clip_text_encoder(inputs).pooler_output.detach()
            else:
                emb = clip_text_encoder(inputs).last_hidden_state.detach()
        target_embeddings.append(emb)

    # Mapping concept embeddings (retain)
    mapping_embeddings = []
    for concept in mapping_concept:
        inputs = encode(concept)
        with torch.no_grad():
            if use_pooler:
                emb = clip_text_encoder(inputs).pooler_output.detach()
            else:
                emb = clip_text_encoder(inputs).last_hidden_state.detach()
        mapping_embeddings.append(emb)

    # Retain prompts - load from CSV file with 'prompt' column
    retain_prompts = []
    retain_embeddings = []

    if retain_csv_path and os.path.exists(retain_csv_path):
        print(f"Loading retain prompts from CSV: {retain_csv_path}")
        df = pd.read_csv(retain_csv_path)

        if 'prompt' not in df.columns:
            raise ValueError(f"CSV file must have a 'prompt' column. Found columns: {df.columns.tolist()}")

        # Load all prompts from CSV
        base_prompts = df['prompt'].dropna().tolist()
        print(f"Loaded {len(base_prompts)} base retain prompts from CSV")

        # Apply prompt augmentation to retain prompts if enabled
        if augment_retain:
            print("Applying prompt augmentation to retain prompts")
            for prompt in base_prompts:
                # Remove "A photo of the " from the beginning if present
                if prompt.startswith("A photo of the "):
                    prompt = prompt[len("A photo of the "):]
                augmented = prompt_augmentation(prompt, augment=True)
                retain_prompts.extend(augmented)
            print(f"Generated {len(retain_prompts)} retain prompts with augmentation")
        else:
            retain_prompts = base_prompts
            print(f"Using {len(retain_prompts)} retain prompts without augmentation")

        # Create cache path for retain embeddings
        cache_dir = os.path.join(output_dir, "cache")
        if is_main:
            os.makedirs(cache_dir, exist_ok=True)

        # Create a cache key based on CSV path, augmentation setting, and pooler setting
        csv_name = os.path.basename(retain_csv_path).replace('.csv', '')
        cache_key = f"{csv_name}_aug{augment_retain}_pooler{use_pooler}"
        cache_path = os.path.join(cache_dir, f"retain_embeddings_{cache_key}.pt")

        # Check if cache exists (before any process tries to create it)
        cache_exists = os.path.exists(cache_path)

        # If cache doesn't exist, main process computes and saves embeddings
        if not cache_exists:
            if is_main:
                print(f"Computing retain embeddings (will cache to: {cache_path})")
                # Create embeddings for retain prompts
                for prompt in tqdm(retain_prompts, desc="Creating retain embeddings"):
                    inputs = encode(prompt)
                    with torch.no_grad():
                        if use_pooler:
                            emb = clip_text_encoder(inputs).pooler_output.detach()
                        else:
                            emb = clip_text_encoder(inputs).last_hidden_state.detach()
                    retain_embeddings.append(emb.squeeze().cpu())  # Store on CPU for caching

                print(f"Caching retain embeddings to: {cache_path}")
                torch.save(retain_embeddings, cache_path)
                print(f"Cached {len(retain_embeddings)} embeddings")

            # Wait for main process to finish creating the cache
            accelerator.wait_for_everyone()

        # All processes load from cache
        if is_main or not cache_exists:
            print(f"Loading cached retain embeddings from: {cache_path}")
        retain_embeddings = torch.load(cache_path, map_location='cpu')
        # Move to correct device
        retain_embeddings = [emb.to(accelerator.device) for emb in retain_embeddings]
        if is_main:
            print(f"Loaded {len(retain_embeddings)} cached embeddings")
    else:
        print("No retain CSV path provided or file not found. Skipping retain loss.")

    print(f"Mapping concepts: {mapping_concept[:2]}...")  # Show first 2
    print(f"Retain prompts: {retain_prompts[:20]} prompts loaded")
    print(f"Retain prompts: {len(retain_prompts)} prompts loaded")

    # Training loop
    criterion = torch.nn.MSELoss()
    losses = []

    # quick_sampler = create_quick_sampler(
    #    accelerator.unwrap_model(model), sampler, resolution, ddim_steps, ddim_eta
    # )
    base = accelerator.unwrap_model(model)
    diag_pipe = FluxPipeline(
        transformer=base,  # <-- THIS is your live transformer
        vae=vae,
        scheduler=noise_scheduler,
        text_encoder=text_encoder_one,
        tokenizer=tokenizer_one,
        text_encoder_2=text_encoder_two,
        tokenizer_2=tokenizer_two,
    )

    #diag_pipe.to(accelerator.device)

    #diag_pipe.transformer.to(device=device, dtype=weight_dtype).eval()
    #diag_pipe.text_encoder.to(device=device, dtype=weight_dtype).eval()
    #diag_pipe.text_encoder_2.to(device=device, dtype=weight_dtype).eval()

    # VAE decode must be fp32
    #diag_pipe.vae.to(device=device, dtype=torch.float32).eval()

    # Make pipeline execution device CUDA
    #diag_pipe = diag_pipe.to(accelerator.device)

    diag_pipe.set_progress_bar_config(disable=True)

    pbar = tqdm(range(max_train_steps), disable=not accelerator.is_local_main_process)

    # Training weights for combining removal and retain losses
    remove_weight = config.get('remove_weight', 1.0)  # Weight for removal loss
    retain_weight = config.get('retain_weight', 0.001)  # Weight for retain loss

    print(f"Loss weights: remove={remove_weight:.3f}, retain={retain_weight:.3f}")

    for iteration in pbar:
        base = accelerator.unwrap_model(model)
        #
        # #optimizer.zero_grad(set_to_none=True)

        vae_config_shift_factor = diag_pipe.vae.config.shift_factor
        vae_config_scaling_factor = diag_pipe.vae.config.scaling_factor
        vae_config_block_out_channels = diag_pipe.vae.config.block_out_channels

        # # Random timestep
        t_enc = torch.randint(ddim_steps, (1,), device=accelerator.device)
        og_num = round((int(t_enc) / ddim_steps) * 100)
        og_num_lim = round((int(t_enc + 1) / ddim_steps) * 1000)
        t_enc_ddpm = torch.randint(og_num, og_num_lim, (1,), device=accelerator.device)
        vae_scale_factor = 2 ** (len(vae_config_block_out_channels))

        num_channels = vae.config.latent_channels

        image_h = 512
        image_w = 512

        latent_h = image_h // vae_scale_factor
        latent_w = image_w // vae_scale_factor
        bsz = 1
        # --- Create dummy latent ---
        model_input = torch.zeros(
            (bsz, num_channels, latent_h, latent_w),
            device=vae.device,
            dtype=weight_dtype,
        )

        # (ESD) start_guidance = 3
        start_guidance = 3
        start_guidance = torch.tensor([start_guidance], device=accelerator.device)
        start_guidance = start_guidance.expand(model_input.shape[0])

        with accelerator.accumulate(model):
            # # REMOVAL LOSS: Push target concepts towards mapping concepts
            # # Use accelerator process index to select a GPU-specific index
            #
            rank = accelerator.process_index
            world_size = accelerator.num_processes

            # All valid indices for THIS GPU only: rank, rank+world_size, ...
            valid_indices = list(range(rank, len(target_embeddings), world_size))

            if len(valid_indices) == 0:
                # Fallback in case there are fewer samples than processes
                concept_idx = rank % len(target_embeddings)
            else:
                # Randomly pick one index from this GPU's slice
                concept_idx = random.choice(valid_indices)

            target_text = target_concepts[concept_idx]
            mapping_text = (
                mapping_concept[concept_idx]
                if concept_idx < len(mapping_concept)
                else mapping_concept[0]
            )

            # Apply prompt augmentation to target if enabled
            # When augmenting, apply the SAME augmentation to both target and mapping
            if augment_target:
                augmented_prompts = prompt_augmentation(target_text, augment=True)

                # Shard augmentation indices per rank as well
                valid_aug_indices = list(range(rank, len(augmented_prompts), world_size))
                if len(valid_aug_indices) == 0:
                    aug_idx = rank % len(augmented_prompts)
                else:
                    aug_idx = random.choice(valid_aug_indices)

                target_text_augmented = augmented_prompts[aug_idx]

                # Apply the SAME augmentation variation to mapping
                augmented_mapping = prompt_augmentation(mapping_text, augment=True)

                # In case augmented_mapping has fewer variants, wrap aug_idx
                if aug_idx >= len(augmented_mapping):
                    aug_idx = aug_idx % len(augmented_mapping)

                mapping_text_augmented = augmented_mapping[aug_idx]

                # Recompute target_emb with the same augmentation
                inputs_aug = encode(target_text_augmented)
                with torch.no_grad():
                    if use_pooler:
                        target_emb = clip_text_encoder(inputs_aug).pooler_output.detach()
                    else:
                        target_emb = clip_text_encoder(inputs_aug).last_hidden_state.detach()
            else:
                target_text_augmented = target_text
                mapping_text_augmented = mapping_text
                target_emb = target_embeddings[concept_idx]

            print(
                f"[Rank {rank} | Device {accelerator.device}] "
                f"idx={concept_idx} | Mapping {target_text_augmented} --> {mapping_text_augmented}"
            )

            with torch.no_grad():
                emb_0, pooled_emb_0, text_ids_0 = compute_text_embeddings(
                    target_text_augmented, text_encoders, tokenizers, accelerator.device
                )
                emb_p, pooled_emb_p, text_ids_p = compute_text_embeddings(
                    target_text_augmented, text_encoders, tokenizers, accelerator.device
                )

            #     # Get text conditioning for Stable Diffusion
            #     emb_p = base.get_learned_conditioning([target_text_augmented])  # target prompt (positive)
            #     emb_n = base.get_learned_conditioning([target_text_augmented])  # target prompt (negative, to be erased)
            #     emb_m = base.get_learned_conditioning([target_text_augmented])  # mapping prompt (what target should map to)
            # # Random timestep for HyperLoRA context
            rank = accelerator.process_index
            world_size = accelerator.num_processes
            #
            # # Timesteps assigned to THIS rank: rank, rank + world_size, ...
            valid_timesteps = torch.arange(rank, hyper_train_steps, world_size, device=accelerator.device)
            if valid_timesteps.numel() == 0:
                # Fallback in case hyper_train_steps < world_size
                rtimestep = int(torch.randint(0, hyper_train_steps, (1,), device=accelerator.device))
            else:
                # Sample index into this rank’s slice
                idx = torch.randint(0, valid_timesteps.numel(), (1,), device=accelerator.device)
                rtimestep = int(valid_timesteps[idx])


            with torch.no_grad():
                with model.hyper.no_lora():
                    z, latent_image_ids = latent_sample(model,
                                                        noise_scheduler,
                                                        1,
                                                        model_input.shape[1],
                                                        512,
                                                        512,
                                                        emb_p.to(accelerator.device),
                                                        pooled_emb_p.to(accelerator.device),
                                                        text_ids_p.to(accelerator.device),
                                                        start_guidance,
                                                        int(ddim_steps))
                    t_ddpm = t_enc_ddpm.to(accelerator.device)  # DON'T cast to bf16

                    e_0 = predict_noise(
                        model, z, emb_0.to(dtype=weight_dtype), pooled_emb_0.to(dtype=weight_dtype), text_ids_0, latent_image_ids,
                        guidance=start_guidance,
                        timesteps=t_ddpm,
                        CPU_only=True,
                    )
                    e_p = predict_noise(
                        model, z, emb_p.to(dtype=weight_dtype), pooled_emb_p.to(dtype=weight_dtype), text_ids_p, latent_image_ids,
                        guidance=start_guidance,
                        timesteps=t_ddpm,
                        CPU_only=True,
                    )

            base.hyper.set_context(target_emb.to(dtype=weight_dtype), torch.tensor([rtimestep], dtype=weight_dtype, device=accelerator.device))
            _, current_timestep = base.hyper.get_context()
            base.hyper.compute_and_cache_loras(target_emb.to(dtype=weight_dtype), current_timestep.to(dtype=weight_dtype))

            e_n = predict_noise(model, z, emb_p.to(dtype=weight_dtype), pooled_emb_p.to(dtype=weight_dtype), text_ids_p, latent_image_ids,
                                guidance=start_guidance, timesteps=t_ddpm, CPU_only=True)
            e_0.requires_grad = False
            e_p.requires_grad = False


            loss_aux = criterion(e_n.to(accelerator.device), e_0.to(accelerato.device) - (
                        negative_guidance * (e_p.to(accelerato.device) - e_0.to(accelerator.device))))


            # with torch.no_grad():
            #     # Generate latent using target prompt
            #     z = quick_sampler(emb_p, start_guidance, start_code, int(t_enc))
            #     # Get noise predictions from original model
            #     e_m = model_orig.apply_model(z, t_enc_ddpm, emb_m)  # mapping (reference) concept
            #     e_p = model_orig.apply_model(z, t_enc_ddpm, emb_p)  # target prompt
            #
            # # Prediction from modified model (with HyperLoRA)
            # _, current_timestep = base.hyper.get_context()
            # base.hyper.compute_and_cache_loras(target_emb, current_timestep)
            # base.hyper.retain_grad_for_cached_lora()
            # e_n = base.apply_model(z, t_enc_ddpm, emb_n)
            #
            # # Loss: push modified output away from target, towards mapping concept
            # e_m.requires_grad_(False)
            # e_p.requires_grad_(False)
            # target = e_m - (negative_guidance * (e_p - e_m))
            # loss_aux = criterion(e_n, target)
            #
            accelerator.backward(loss_aux)

            # --- use cached LoRA grads instead of live-tensor grads ---
            grads_flat_t = base.hyper.flatten_cached_grads_from_cache()
            if grads_flat_t is None:
                raise RuntimeError(
                    "No gradients found in cached LoRA tensors. Ensure cache is built with graph intact and retain_grad() was called.")

            # Target step: Δθ ≈ -lr * g_t  (keep target detached)
            grads_flat_t = (-1.0 * internal_lr) * grads_flat_t.detach()

            #for p in trainable_params:
            #    if p.grad is not None:
            #        p.grad = None

            _, current_timestep = accelerator.unwrap_model(model).hyper.get_context()
            base.hyper.set_context(target_emb, current_timestep)
            base.hyper.compute_and_cache_loras(target_emb, current_timestep)
            tensors_flat_t = base.hyper.flatten_cached_from_cache()

            base.hyper.set_context(target_emb, current_timestep + 1)
            base.hyper.compute_and_cache_loras(target_emb, current_timestep + 1)
            tensors_flat_t1 = base.hyper.flatten_cached_from_cache()

            # Match the SGD step: (θ_{t+1} - θ_t) ≈ -lr * g_t
            delta_live = tensors_flat_t1 - tensors_flat_t
            loss_remove = remove_weight * criterion(delta_live, grads_flat_t)
            accelerator.backward(loss_remove)

            if len(retain_embeddings) > 0:
                # Sample multiple retain concepts
                num_retain_samples = min(10, len(retain_embeddings))
                sampled_retain_embs = random.sample(retain_embeddings, num_retain_samples)

                # Batch process retain concepts
                batch_retain_embs = (
                    torch.stack(sampled_retain_embs, dim=0)
                        .to(device=accelerator.device, dtype=torch.bfloat16)
                )
                hyper = base.hyper
                batch_prompts = batch_retain_embs.repeat(hyper_train_steps // num_retain_samples, 1)
                B = batch_prompts.shape[0]
                perm = torch.randperm(B, device=batch_prompts.device)
                batch_prompts = batch_prompts[perm]

                # Compute LoRAs at t=0
                dtype = next(hyper.parameters()).dtype  # hyper’s param dtype (bf16 if you casted it)

                hyper.compute_and_cache_loras(
                    batch_prompts.to(dtype=dtype),
                    torch.zeros(B, device=accelerator.device, dtype=dtype),
                )

                tensors_flat_t0 = hyper.flatten_cached_from_cache()

                #Compute LoRAs at t=1, 2, 3, ... B
                dtype = next(hyper.parameters()).dtype

                t_ = (torch.arange(B, device=accelerator.device, dtype=dtype) % B) + 1
                hyper.compute_and_cache_loras(
                    batch_prompts.to(dtype=dtype),
                    t_,
                )
                tensors_flat_t1 = hyper.flatten_cached_from_cache()

                #Loss: minimize change in LoRA weights across timesteps
                delta = tensors_flat_t1 - tensors_flat_t0
                loss_retain = retain_weight * delta.pow(2).mean()
            else:
                loss_retain = torch.tensor(0.0, device=accelerator.device)
                #loss_retain = torch.tensor(0.0, device=accelerator.device)
            accelerator.backward(loss_retain)

            loss_remove_log = loss_remove.clone().detach()
            loss_retain_log = loss_retain.clone().detach()

            # Optimizer step
            if accelerator.sync_gradients:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()

        # Gather loss across devices
        with torch.no_grad():
            loss_retain_reduced = accelerator.gather(loss_retain_log).mean()
            loss_remove_reduced = accelerator.gather(loss_remove_log).mean()

        losses.append(float(loss_remove_reduced.item() + loss_retain_reduced.item()))

        if is_main and use_wandb:
            wandb.log({
                "loss_retain": float(loss_retain_reduced.item()),
                "loss_remove": float(loss_remove_reduced.item())
            }, step=iteration)

        if is_main:
            pbar.set_postfix({
                "retain": f"{float(loss_retain_reduced.item()):.6f}",
                "remove": f"{float(loss_remove_reduced.item()):.6f}"
            })

        # Generate sample images periodically
        if is_main and use_wandb and (iteration + 1) % 50 == 0:
            # Generate images for diagnostic prompts from config
            for diag_idx, diag_prompt in enumerate(diagnostic_prompts):

                # 1) Compute diag_emb on GPU only if you need HyperLoRA context from CLIP.
                # If clip_text_encoder is huge, consider moving it to GPU only for this block.
                inputs_diag = encode(diag_prompt)
                with torch.no_grad():
                    if use_pooler:
                        diag_emb = clip_text_encoder(inputs_diag).pooler_output.detach()
                    else:
                        diag_emb = clip_text_encoder(inputs_diag).last_hidden_state.detach()

                diag_time_steps = [0, hyper_train_steps // 2, hyper_train_steps]

                device = accelerator.device
                weight_dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

                # 2) Ensure pipeline uses the *live* transformer (no copies)
                base = accelerator.unwrap_model(model)
                diag_pipe.transformer = base  # make sure pipe uses current model

                # 3) Move ONLY what you need to GPU for diagnostics (encoders+VAE), then move back
                #    Avoid diag_pipe.to(device) if you're tight on VRAM; move components explicitly.

                diag_seed = 12345  # fixed so noise identical across h_step
                imgs_per_prompt = []
                for h_step in diag_time_steps:
                    h_step_tensor = torch.tensor([h_step], device=device)

                    # Enable these if you want hyper-time to change the result
                    base.hyper.set_context(diag_emb.to(dtype=weight_dtype), h_step_tensor.to(dtype=weight_dtype))
                    base.hyper.compute_and_cache_loras(diag_emb.to(dtype=weight_dtype), h_step_tensor.to(dtype=weight_dtype))

                    diag_pipe.text_encoder.to(device=device, dtype=weight_dtype).eval()
                    diag_pipe.text_encoder_2.to(device=device, dtype=weight_dtype).eval()
                    diag_pipe.vae.to(device=device, dtype=torch.float32).eval()

                    generator = torch.Generator(device=device).manual_seed(diag_seed)

                    with torch.no_grad():
                        # IMPORTANT: avoid internal VAE decode to prevent bf16->fp32 mismatch + extra VRAM
                        diag_pipe.vae.to(device=device, dtype=torch.bfloat16)
                        imgs = diag_pipe(
                            prompt=diag_prompt,
                            guidance_scale=guidance_scale,
                            num_inference_steps=50,
                            height=resolution,
                            width=resolution,
                            generator=generator,
                            max_sequence_length=256,
                        ).images

                    imgs_per_prompt.append(imgs)

                # 5) Move encoders/VAE back to CPU to free VRAM for training
                diag_pipe.text_encoder.to("cpu")
                diag_pipe.text_encoder_2.to("cpu")
                diag_pipe.vae.to("cpu")
                torch.cuda.empty_cache()

                # 6) Log a single concatenated image to W&B
                if len(imgs_per_prompt) > 0:
                    row_tensors = []

                    for imgs in imgs_per_prompt:
                        if imgs is None:
                            continue

                        # Take the first image (assumed to be PIL.Image)
                        img = imgs[0]
                        img = to_tensor(img).clamp(0, 1)
                        row_tensors.append(img)

                    if len(row_tensors) > 0:
                        # Concatenate horizontally to form a row: (C, H, sum_W)
                        row = torch.cat(row_tensors, dim=2)

                        # Clean prompt for wandb key (remove spaces and special chars)
                        safe_key = diag_prompt.replace(" ", "_").replace(",", "")[:50]

                        wandb.log(
                            {
                                f"diagnostic_{diag_idx}_{safe_key}": wandb.Image(
                                    to_pil_image(row),
                                    caption=f"{diag_prompt} | hyper steps: {diag_time_steps}",
                                )
                            },
                            step=iteration,
                        )


        # Save model
        accelerator.wait_for_everyone()
        if is_main and ((iteration % 100 == 0) or (iteration == max_train_steps - 1)):
            print(f"Final loss: {losses[-1]:.6f}")
            print(f"Average loss: {sum(losses) / len(losses):.6f}")

            # Create output directory
            os.makedirs(output_dir, exist_ok=True)
            os.makedirs(final_save_path, exist_ok=True)

            # Save LoRA weights
            lora_state_dict = {}
            # model_unwrapped = accelerator.unwrap_model(model)
            # for name, param in model_unwrapped.model.diffusion_model.named_parameters():
            #    if param.requires_grad:
            #        lora_state_dict[name] = param.detach().cpu().clone()

            lora_path = os.path.join(final_save_path, f"hyper_lora_{iteration}.pth")
            accelerator.save(lora_state_dict, lora_path)
            print(f"Model saved to: {lora_path}")

        # Save config
        config_save = {
            "config_name": config_name,
            "concepts": concepts,
            "mapping_concept": mapping_concept,
            "retain_csv_path": retain_csv_path,
            "augment_target": augment_target,
            "augment_retain": augment_retain,
            "num_retain_prompts": len(retain_prompts),
            "rank": rank,
            "learning_rate": learning_rate,
            "max_train_steps": max_train_steps,
            "hyper_train_steps": hyper_train_steps,
            "final_loss": losses[-1],
            "average_loss": sum(losses) / len(losses),
        }

        with open(os.path.join(final_save_path, "train_config.json"), "w") as f:
            json.dump(config_save, f, indent=2)

    if is_main and use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
