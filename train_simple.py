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
import copy
from pathlib import Path
from functools import partial

import pandas as pd
import torch
import torch.nn as nn
import wandb
import yaml
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration, set_seed as hf_set_seed
from torchvision.transforms.functional import to_pil_image
from transformers import CLIPTextModel, CLIPTokenizer
from tqdm import tqdm

from hyper_lora import HyperLoRALinear, HypernetworkManager, inject_hyper_lora
from ldm.models.diffusion.ddimcopy import DDIMSampler
from sampling import sample_model
from utils import load_model_from_config, print_trainable_parameters

# Lazy imports for NV-Embed (only loaded when needed)
_nv_embed_module = None

def _load_nv_embed_module():
    global _nv_embed_module
    if _nv_embed_module is None:
        import nv_embed_utils
        _nv_embed_module = nv_embed_utils
    return _nv_embed_module


class HyperCache:
    """
    Cache for embeddings used as HyperLoRA context.
    Supports NV-Embed, CLIP, or any embedding function.
    All embeddings stored on CPU, moved to device on access.
    """

    def __init__(
        self,
        prompts: list = None,
        embed_fn=None,
        device: torch.device = None,
        batch_size: int = 8,
        embed_model_name: str = "unknown",
    ):
        self.prompts = []
        self.prompt_to_idx = {}
        self.embeddings = None
        self.dirty = False
        self.embed_model_name = embed_model_name

        if prompts and embed_fn is not None:
            print(f"[HyperCache] Computing {embed_model_name} embeddings for {len(prompts)} prompts...")
            self.prompts = list(prompts)
            self.prompt_to_idx = {p: i for i, p in enumerate(self.prompts)}

            # Compute embeddings in batches
            all_embeddings = []
            for i in range(0, len(prompts), batch_size):
                batch = prompts[i:i+batch_size]
                batch_embs = [embed_fn(p).cpu() for p in batch]
                all_embeddings.extend(batch_embs)

            self.embeddings = torch.cat([e if e.dim() == 2 else e.unsqueeze(0) for e in all_embeddings], dim=0)
            self._print_memory_usage()

    def get(self, prompt: str, device: torch.device) -> torch.Tensor:
        """Get embedding for prompt, moved to device."""
        idx = self.prompt_to_idx[prompt]
        return self.embeddings[idx:idx+1].to(device)

    def get_by_idx(self, idx: int, device: torch.device) -> torch.Tensor:
        """Get embedding by index, moved to device."""
        return self.embeddings[idx:idx+1].to(device)

    def get_batch(self, prompts: list, device: torch.device) -> torch.Tensor:
        """Get batch of embeddings, moved to device."""
        indices = [self.prompt_to_idx[p] for p in prompts]
        return self.embeddings[indices].to(device)

    def get_batch_by_idx(self, indices: list, device: torch.device) -> torch.Tensor:
        """Get batch of embeddings by indices, moved to device."""
        return self.embeddings[indices].to(device)

    def sample_batch(self, n: int, device: torch.device) -> torch.Tensor:
        """Sample n random embeddings, moved to device."""
        if self.embeddings is None or len(self.prompts) == 0:
            return None
        n = min(n, len(self.prompts))
        indices = random.sample(range(len(self.prompts)), n)
        return self.embeddings[indices].to(device)

    def has(self, prompt: str) -> bool:
        """Check if prompt is cached."""
        return prompt in self.prompt_to_idx

    def add(self, prompt: str, embedding: torch.Tensor):
        """Add embedding to cache (stored on CPU)."""
        if prompt in self.prompt_to_idx:
            return
        idx = len(self.prompts)
        self.prompts.append(prompt)
        self.prompt_to_idx[prompt] = idx
        emb = embedding.cpu()
        if emb.dim() == 1:
            emb = emb.unsqueeze(0)
        if self.embeddings is None:
            self.embeddings = emb
        else:
            self.embeddings = torch.cat([self.embeddings, emb], dim=0)
        self.dirty = True

    def __len__(self) -> int:
        return len(self.prompts)

    def __contains__(self, prompt: str) -> bool:
        return self.has(prompt)

    def _print_memory_usage(self):
        if self.embeddings is None:
            print(f"[HyperCache] Empty cache")
            return
        mem_mb = self.embeddings.element_size() * self.embeddings.nelement() / (1024 ** 2)
        print(f"[HyperCache] Memory: {mem_mb:.2f}MB ({len(self.prompts)} prompts)")

    def save(self, path: str):
        """Save cache to disk."""
        dirname = os.path.dirname(path)
        if dirname:
            os.makedirs(dirname, exist_ok=True)
        data = {
            'prompts': self.prompts,
            'embeddings': self.embeddings,
            'version': 1,
        }
        torch.save(data, path)
        print(f"[HyperCache] Saved to {path}")

    @classmethod
    def load(cls, path: str, expected_prompts: list = None):
        """Load cache from disk."""
        print(f"[HyperCache] Loading from {path}...")
        data = torch.load(path, map_location='cpu', weights_only=False)

        if expected_prompts is not None:
            if set(data['prompts']) != set(expected_prompts):
                print(f"[HyperCache] Prompt mismatch, cache invalid")
                return None

        instance = object.__new__(cls)
        instance.prompts = data['prompts']
        instance.prompt_to_idx = {p: i for i, p in enumerate(instance.prompts)}
        instance.embeddings = data['embeddings']
        instance.dirty = False
        instance._print_memory_usage()
        return instance


class CombinedCFGModel:
    """Wrapper that uses the same model but toggles LoRA for unconditional/reference passes."""

    def __init__(self, model):
        self.model = model
        self.device = model.device

    def apply_model(self, x, t, c):
        # When DDIMSampler uses guidance, it concatenates [uncond, cond] inputs
        # We split and route to the same model with LoRA toggled
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

        # Route unconditional to model without LoRA
        with self.model.hyper.no_lora():
            out_uncond = self.model.apply_model(x_uncond, t_uncond, c_uncond)
        
        # Route conditional to model with LoRA
        out_cond = self.model.apply_model(x_cond, t_cond, c_cond)

        return torch.cat([out_uncond, out_cond], dim=0)

    def get_learned_conditioning(self, prompts):
        return self.model.get_learned_conditioning(prompts)

    def decode_first_stage(self, z):
        return self.model.decode_first_stage(z)

    def eval(self):
        self.model.eval()
        return self

    def __getattr__(self, name):
        return getattr(self.model, name)


def prompt_augmentation(content, augment=True, celebrity=False):
    """Generate augmented prompts for a given concept."""
    if not augment:
        return [content]

    if celebrity:
        # celebrity augmentation
        prompts = [
            "A photo of {}".format(content),
            "A portrait of {}".format(content),
            "A headshot of {}".format(content),
            "A close-up photo of {}".format(content),
            "A professional photo of {}".format(content),
            "{} smiling".format(content),
            "{} looking at the camera".format(content),
            "A candid photo of {}".format(content),
            "{} at a press conference".format(content),
            "{} at an interview".format(content),
            "{} on the red carpet".format(content),
            "{} at a premiere".format(content),
            "{} at an award ceremony".format(content),
            "{} at a public event".format(content),
            "{} giving a speech".format(content),
            "A black and white photo of {}".format(content),
            "A vintage photo of {}".format(content),
            "A studio portrait of {}".format(content),
            "{} in formal attire".format(content),
            "{} in casual clothes".format(content),
            "A magazine cover featuring {}".format(content),
            "A paparazzi photo of {}".format(content),
            "{} waving to fans".format(content),
            "{} signing autographs".format(content),
            "A selfie of {}".format(content),
            "{} on a talk show".format(content),
            "{} being interviewed".format(content),
            "A painting of {}".format(content),
            "An oil portrait of {}".format(content),
            "A digital art of {}".format(content),
            "A sketch of {}".format(content),
            "A caricature of {}".format(content),
            "{} in a movie scene".format(content),
            "{} on set".format(content),
            "A promotional photo of {}".format(content),
            "{} posing for photographers".format(content),
            "An official photo of {}".format(content),
            "{} at a charity event".format(content),
            "{} at a film festival".format(content),
            "A young {}".format(content),
        ]
    else:
        # object augmentation
        prompts = [
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
        description="Simplified HyperLoRA Training"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML configuration file",
    )
    parser.add_argument(
        "--use_huge",
        action="store_true",
        default=False,
        help="Use largest CLIP model (ViT-G/14, 1280 dim) instead of ViT-L/14 (768 dim)",
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
    learning_rate_remove = config.get('learning_rate_remove', 1e-5)
    learning_rate_retain = config.get('learning_rate_retain', 1e-5)
    max_train_steps = config.get('max_train_steps', 120)
    hyper_train_steps = config.get('hyper_train_steps', 500)
    rank_lora = config.get('rank', 1) # named rank_lora to avoid confusion with proc rank
    lora_alpha = config.get('lora_alpha', 8)
    internal_size = config.get('internal_size', 100)
    seed = config.get('seed', 2024)
    resolution = config.get('resolution', 512)
    use_orig_concat = config.get('use_orig_concat', False)
    gradient_accumulation_steps = config.get('gradient_accumulation_steps', 1)
    
    # Multi-concept configuration
    concepts = config.get('concepts', [])
    mapping_concept = config.get('mapping_concept', [])
    retain_csv_path = config.get('retain_csv_path', None)
    
    # Augmentation flags
    augment_target = config.get('augment_target', True)
    augment_retain = config.get('augment_retain', False)
    celebrity_mode = config.get('celebrity_mode', False)
    use_huge = config.get('use_huge', False)

    # Embedding model configuration: "clip", "clip_huge", or "nv_embed"
    embedding_model = config.get('embedding_model', 'clip')
    use_pooler = config.get('use_pooler', True)

    # Retain balancing parameters
    retain_steps_per_remove = config.get('retain_steps_per_remove', 1)
    retain_batch_size = config.get('retain_batch_size', min(64, retain_steps_per_remove))
    learning_rate_retain = learning_rate_retain
    
    # Paths
    output_dir = config.get('output_dir', './output')
    final_save_path = config.get('final_save_path', './saved_model/LoRA_fusion_model')
    pretrained_model_path = config.get('pretrained_model_name_or_path', './models/sd-v1-4.ckpt')
    model_config_path = config.get('model_config', './configs/stable-diffusion/v1-inference.yaml')
    
    # Training settings
    ddim_steps = 50
    ddim_eta = 0.0
    negative_guidance = config.get('negative_guidance', 2.0)
    guidance_scale = config.get('guidance_scale', 7.5)
    start_guidance = config.get('guidance_scale', 9.0)
    internal_lr = config.get('internal_lr', 1e-4)
    
    diagnostic_prompts = config.get('diagnostic_prompts', [])
    if not diagnostic_prompts:
        diagnostic_prompts = [
            f"a photo of {concepts[0]}" if concepts else "a photo of a person",
            "a photo of a cat",
            "a photo of a car"
        ]
    
    print(f"Training steps: {max_train_steps}")
    print(f"Hypernetwork steps: {hyper_train_steps}")
    print(f"Learning rate (remove): {learning_rate_remove}")
    print(f"Learning rate (retain): {learning_rate_retain}")
    print(f"Retain steps per remove: {retain_steps_per_remove}")
    print(f"Retain batch size: {retain_batch_size}")
    print(f"LoRA rank: {rank_lora}")
    print(f"LoRA alpha: {lora_alpha}")
    print(f"Target concepts: {len(concepts)}")
    print("=" * 48)
    
    if seed is not None:
        hf_set_seed(seed)
    
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
    
    use_wandb = config.get('report_to') == 'wandb'
    if is_main and use_wandb:
        wandb.init(
            project="UnGuide",
            name=f"{config_name}_training",
            config=config
        )
    
    # Load model (only one)
    model = load_model_from_config(
        model_config_path, pretrained_model_path, accelerator.device
    )
    
    # Freeze backbone
    for p in model.model.diffusion_model.parameters():
        p.requires_grad = False
    
    # Setup HyperLoRA
    model.hyper = HypernetworkManager()

    # Determine embedding dimension based on embedding_model config
    if embedding_model == 'nv_embed':
        nv_embed_mod = _load_nv_embed_module()
        clip_size = nv_embed_mod.NV_EMBED_DIM  # 4096
    elif embedding_model == 'clip_huge':
        clip_size = 1280
    else:  # default: clip
        clip_size = 768 if use_pooler else 512

    target_modules = ["attn2.to_k", "attn2.to_v"]
    
    hyper_lora_factory = partial(
        HyperLoRALinear,
        clip_size=clip_size,
        rank=rank_lora,
        alpha=lora_alpha,
        train_steps=hyper_train_steps,
        use_orig_concat=use_orig_concat,
        internal_size=internal_size,
    )
    
    hyper_lora_layers = inject_hyper_lora(
        model.model.diffusion_model, target_modules, hyper_lora_factory
    )
    
    for layer_name, layer in hyper_lora_layers:
        layer.set_parent_model(model)
    
    # Setup optimizer
    trainable_params = list(filter(lambda p: p.requires_grad, model.model.diffusion_model.parameters()))
    
    if is_main:
        print(f"Total trainable parameter tensors: {len(trainable_params)}")
        print_trainable_parameters(model)
    
    optimizer_remove = torch.optim.Adam(trainable_params, lr=learning_rate_remove)
    optimizer_retain = torch.optim.Adam(trainable_params, lr=learning_rate_retain)

    gamma = config.get('gamma', 0.9)
    step_size = config.get('step_size', 300)

    drop_lr_on_plateau = config.get('drop_lr_on_plateau', False)
    if drop_lr_on_plateau:
        plateau_factor = config.get('plateau_factor', 0.1)
        plateau_patience_remove = config.get('plateau_patience_remove', 10)
        plateau_patience_retain = config.get('plateau_patience_retain', 10)

        scheduler_remove = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer_remove,
            mode='min',
            factor=plateau_factor,
            patience=plateau_patience_remove,
        )
        scheduler_retain = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer_retain,
            mode='min',
            factor=plateau_factor,
            patience=plateau_patience_retain,
        )
        if is_main:
            print(f"Using separate ReduceLROnPlateau schedulers:")
            print(f"  Remove: lr={learning_rate_remove}, factor={plateau_factor}, patience={plateau_patience_remove}")
            print(f"  Retain: lr={learning_rate_retain}, factor={plateau_factor}, patience={plateau_patience_retain}")
    else:
        scheduler_remove = torch.optim.lr_scheduler.MultiStepLR(
            optimizer_remove, milestones=[step_size], gamma=gamma
        )
        scheduler_retain = torch.optim.lr_scheduler.MultiStepLR(
            optimizer_retain, milestones=[step_size], gamma=gamma
        )
        if is_main:
            print(f"Using MultiStepLR schedulers (step_size={step_size}, gamma={gamma})")
    
    model, optimizer_remove, optimizer_retain = accelerator.prepare(model, optimizer_remove, optimizer_retain)
    
    for layer_name, layer in hyper_lora_layers:
        layer.set_parent_model(accelerator.unwrap_model(model))
        accelerator.unwrap_model(model).hyper.add_hyperlora(layer_name, layer.hyper_lora)
    
    sampler = DDIMSampler(accelerator.unwrap_model(model))

    # Setup embedding model based on config
    nv_embed_model = None
    nv_embed_tokenizer = None
    clip_text_encoder = None
    tokenizer = None
    use_open_clip = False

    if embedding_model == 'nv_embed':
        nv_embed_mod = _load_nv_embed_module()
        print(f"Loading {nv_embed_mod.NV_EMBED_MODEL_NAME} for HyperLoRA context embeddings...")
        nv_embed_model, nv_embed_tokenizer = nv_embed_mod.load_nv_embed_model(
            accelerator.device, torch.float16
        )
        # Still need CLIP tokenizer for the diffusion model conditioning
        tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
        clip_text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14").to(accelerator.device).eval()

        def get_embedding(text: str):
            with torch.no_grad():
                return nv_embed_mod.compute_nv_embed(
                    [text], nv_embed_model, nv_embed_tokenizer, accelerator.device
                ).detach()

        embed_model_name = "nv_embed"

    elif embedding_model == 'clip_huge':
        import open_clip
        print("Using HUGE CLIP model: ViT-bigG-14 (1280 dim) via open_clip")
        clip_model, _, _ = open_clip.create_model_and_transforms('ViT-bigG-14', pretrained='laion2b_s39b_b160k')
        clip_text_encoder = clip_model.to(accelerator.device).eval()
        tokenizer = open_clip.get_tokenizer('ViT-bigG-14')
        use_open_clip = True

        def get_embedding(text: str):
            with torch.no_grad():
                tokens = tokenizer(text).to(accelerator.device)
                return clip_text_encoder.encode_text(tokens).detach()

        embed_model_name = "clip_huge"

    else:  # default: clip
        print("Using standard CLIP model: ViT-L/14 (768 dim)")
        tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
        clip_text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14").to(accelerator.device).eval()

        def get_embedding(text: str):
            with torch.no_grad():
                inputs = tokenizer(
                    text,
                    max_length=tokenizer.model_max_length,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt",
                ).to(accelerator.device).input_ids
                if use_pooler:
                    return clip_text_encoder(inputs).pooler_output.detach()
                else:
                    return clip_text_encoder(inputs).last_hidden_state.detach()

        embed_model_name = "clip"

    # Print embedding dimensionality for verification
    target_concepts = [c for c in concepts]
    if target_concepts:
        test_emb = get_embedding(target_concepts[0])
        print(f"Embedding shape ({embed_model_name}): {test_emb.shape} (first concept: '{target_concepts[0]}')")

    # Build list of all prompts to cache
    all_prompts_to_cache = []

    # Add augmented target prompts
    all_augmented_targets = []
    for concept in target_concepts:
        if augment_target:
            augmented = prompt_augmentation(concept, augment=True, celebrity=celebrity_mode)
            all_augmented_targets.extend(augmented)
        else:
            all_augmented_targets.append(concept)
    all_prompts_to_cache.extend(all_augmented_targets)

    # Add augmented mapping prompts
    all_augmented_mappings = []
    for concept in mapping_concept:
        if augment_target:
            augmented = prompt_augmentation(concept, augment=True, celebrity=celebrity_mode)
            all_augmented_mappings.extend(augmented)
        else:
            all_augmented_mappings.append(concept)
    all_prompts_to_cache.extend(all_augmented_mappings)

    # Add diagnostic prompts
    all_prompts_to_cache.extend(diagnostic_prompts)

    retain_prompts = []
    if retain_csv_path and os.path.exists(retain_csv_path):
        print(f"Loading retain prompts from CSV: {retain_csv_path}")
        df = pd.read_csv(retain_csv_path)
        if 'prompt' not in df.columns:
            raise ValueError(f"CSV file must have a 'prompt' column. Found columns: {df.columns.tolist()}")

        base_prompts = df['prompt'].dropna().tolist()
        if augment_retain:
            for prompt in base_prompts:
                if prompt.startswith("A photo of the "):
                    prompt = prompt[len("A photo of the "):]
                augmented = prompt_augmentation(prompt, augment=True)
                retain_prompts.extend(augmented)
        else:
            retain_prompts = base_prompts
        all_prompts_to_cache.extend(retain_prompts)

    # Remove duplicates while preserving order
    seen = set()
    unique_prompts = []
    for p in all_prompts_to_cache:
        if p not in seen:
            seen.add(p)
            unique_prompts.append(p)
    all_prompts_to_cache = unique_prompts

    # Setup cache directory and path
    cache_dir = os.path.join(output_dir, "cache")
    if is_main:
        os.makedirs(cache_dir, exist_ok=True)

    cache_name = concepts[0].replace(' ', '_').replace(',', '')[:30] if concepts else 'default'
    cache_path = os.path.join(cache_dir, f"hyper_cache_{cache_name}.pt")

    # Load or create HyperCache
    hyper_cache = None
    if os.path.exists(cache_path):
        hyper_cache = HyperCache.load(cache_path, expected_prompts=all_prompts_to_cache)

    if hyper_cache is None:
        if is_main:
            hyper_cache = HyperCache(
                prompts=all_prompts_to_cache,
                embed_fn=get_embedding,
                device=accelerator.device,
                batch_size=8,
                embed_model_name=embed_model_name,
            )
            hyper_cache.save(cache_path)
        accelerator.wait_for_everyone()
        if not is_main:
            hyper_cache = HyperCache.load(cache_path)

    print(f"[HyperCache] {len(hyper_cache)} prompts cached ({embed_model_name})")
    
    criterion = torch.nn.MSELoss()
    losses = []
    
    quick_sampler = create_quick_sampler(
        accelerator.unwrap_model(model), sampler, resolution, ddim_steps, ddim_eta
    )
    
    pbar = tqdm(range(max_train_steps), disable=not accelerator.is_local_main_process)
    
    remove_weight = config.get('remove_weight', 1.0)
    retain_weight = config.get('retain_weight', 0.001)
    
    for iteration in pbar:
        base = accelerator.unwrap_model(model)
        
        t_enc = torch.randint(ddim_steps, (1,), device=accelerator.device)
        og_num = round((int(t_enc) / ddim_steps) * 100)
        og_num_lim = round((int(t_enc + 1) / ddim_steps) * 1000)
        t_enc_ddpm = torch.randint(og_num, og_num_lim, (1,), device=accelerator.device)
        start_code = torch.randn((1, 4, resolution // 8, resolution // 8), device=accelerator.device)

        # Zero gradients at start of iteration (fix: mirrors train_flux_simple.py pattern)
        optimizer_remove.zero_grad(set_to_none=True)
        optimizer_retain.zero_grad(set_to_none=True)

        with accelerator.accumulate(model):
            rank_proc = accelerator.process_index
            world_size = accelerator.num_processes
            valid_indices = list(range(rank_proc, len(target_concepts), world_size))

            if len(valid_indices) == 0:
                concept_idx = rank_proc % len(target_concepts)
            else:
                concept_idx = random.choice(valid_indices)

            target_text = target_concepts[concept_idx]
            mapping_text = mapping_concept[concept_idx] if concept_idx < len(mapping_concept) else mapping_concept[0]

            if augment_target:
                augmented_prompts = prompt_augmentation(target_text, augment=True, celebrity=celebrity_mode)
                valid_aug_indices = list(range(rank_proc, len(augmented_prompts), world_size))
                aug_idx = random.choice(valid_aug_indices) if len(valid_aug_indices) > 0 else rank_proc % len(augmented_prompts)
                target_text_augmented = augmented_prompts[aug_idx]
                augmented_mapping = prompt_augmentation(mapping_text, augment=True, celebrity=celebrity_mode)
                mapping_text_augmented = augmented_mapping[aug_idx % len(augmented_mapping)]

                print(target_text_augmented, ' -> ', mapping_text_augmented)

            else:
                target_text_augmented = target_text
                mapping_text_augmented = mapping_text

            # Get embedding from cache (moves from CPU to device)
            target_emb = hyper_cache.get(target_text_augmented, accelerator.device)

            with torch.no_grad():
                emb_p = base.get_learned_conditioning([target_text_augmented])
                emb_n = base.get_learned_conditioning([target_text_augmented])
                emb_m = base.get_learned_conditioning([mapping_text_augmented])
            
            valid_timesteps = torch.arange(rank_proc, hyper_train_steps, world_size, device=accelerator.device)
            rtimestep = int(valid_timesteps[torch.randint(0, valid_timesteps.numel(), (1,))]) if valid_timesteps.numel() > 0 else int(torch.randint(0, hyper_train_steps, (1,), device=accelerator.device))
            
            base.hyper.set_context(target_emb, torch.tensor([rtimestep], device=accelerator.device))
            _, current_timestep = base.hyper.get_context()
            base.hyper.compute_and_cache_loras(target_emb, current_timestep)
            
            with torch.no_grad():
                # Use base model without LoRA for reference outputs
                with base.hyper.no_lora():
                    #TODO: SAMPLING LATENT FROM MODEL WITHOUT LORA
                    z = quick_sampler(emb_p, start_guidance, start_code, int(t_enc))
                    e_m = base.apply_model(z, t_enc_ddpm, emb_m)
                    e_p = base.apply_model(z, t_enc_ddpm, emb_p)
            
            base.hyper.set_context(target_emb, current_timestep)
            base.hyper.compute_and_cache_loras(target_emb, current_timestep)
            base.hyper.retain_grad_for_cached_lora()
            e_n = base.apply_model(z, t_enc_ddpm, emb_n)
            
            target = e_m - (negative_guidance * (e_p - e_m))
            loss_aux = criterion(e_n, target)
            accelerator.backward(loss_aux)

            grads_flat_t = base.hyper.flatten_cached_grads_from_cache()
            if grads_flat_t is None:
                raise RuntimeError("No gradients found in cached LoRA tensors.")

            grads_flat_t = (-1.0 * internal_lr) * grads_flat_t.detach()

            base.hyper.set_context(target_emb, current_timestep)
            base.hyper.compute_and_cache_loras(target_emb, current_timestep)
            tensors_flat_t = base.hyper.flatten_cached_from_cache()
            
            base.hyper.set_context(target_emb, current_timestep + 1)
            base.hyper.compute_and_cache_loras(target_emb, current_timestep + 1)
            tensors_flat_t1 = base.hyper.flatten_cached_from_cache()
            
            delta_live = tensors_flat_t1 - tensors_flat_t
            loss_remove = remove_weight * criterion(delta_live, grads_flat_t)
            accelerator.backward(loss_remove)

            loss_remove_log = loss_remove.clone().detach()

            if accelerator.sync_gradients:
                #print("sync_gradients:", accelerator.sync_gradients)
                optimizer_remove.step()
                optimizer_remove.zero_grad(set_to_none=True)
                if drop_lr_on_plateau:
                    scheduler_remove.step(loss_remove.detach())
                else:
                    scheduler_remove.step()

            loss_retain_total = torch.tensor(0.0, device=accelerator.device)
            if len(retain_prompts) > 0:
                for retain_step in range(retain_steps_per_remove):
                    num_retain_samples = min(retain_batch_size, len(retain_prompts))
                    sampled_retain_prompts = random.sample(retain_prompts, num_retain_samples)
                    batch_retain_embs = hyper_cache.get_batch(sampled_retain_prompts, accelerator.device)

                    hyper = base.hyper
                    batch_prompts = batch_retain_embs.repeat(max(1, hyper_train_steps // num_retain_samples), 1)
                    B = batch_prompts.shape[0]
                    perm = torch.randperm(B, device=batch_prompts.device)
                    batch_prompts = batch_prompts[perm]

                    hyper.compute_and_cache_loras(batch_prompts, torch.zeros(B, device=accelerator.device))
                    tensors_flat_t0 = hyper.flatten_cached_from_cache()

                    t_ = (torch.arange(B, device=accelerator.device) % B) + 1
                    hyper.compute_and_cache_loras(batch_prompts, t_)
                    tensors_flat_t1 = hyper.flatten_cached_from_cache()

                    delta = tensors_flat_t1 - tensors_flat_t0
                    loss_retain = retain_weight * delta.pow(2).mean()
                    loss_retain_total = loss_retain_total + loss_retain.detach()

                    accelerator.backward(loss_retain)

                    if accelerator.sync_gradients:
                        optimizer_retain.step()
                        optimizer_retain.zero_grad(set_to_none=True)

                if accelerator.sync_gradients:
                    loss_retain_total /= retain_steps_per_remove
                    if drop_lr_on_plateau:
                        scheduler_retain.step(loss_retain_total)
                    else:
                        scheduler_retain.step()

            loss_retain_log = loss_retain_total / max(1, retain_steps_per_remove)
        
        with torch.no_grad():
            loss_retain_reduced = accelerator.gather(loss_retain_log).mean()
            loss_remove_reduced = accelerator.gather(loss_remove_log).mean()
        losses.append(float(loss_remove_reduced.item() + loss_retain_reduced.item()))

        if is_main and use_wandb:
            current_lr_remove = optimizer_remove.param_groups[0]['lr']
            current_lr_retain = optimizer_retain.param_groups[0]['lr']
            wandb.log({
                "loss_retain": float(loss_retain_reduced.item()),
                "loss_remove": float(loss_remove_reduced.item()),
                "learning_rate_remove": current_lr_remove,
                "learning_rate_retain": current_lr_retain,
                "retain_steps_per_remove": retain_steps_per_remove,
            }, step=iteration)
        
        if is_main:
            pbar.set_postfix({"retain": f"{float(loss_retain_reduced.item()):.6f}", "remove": f"{float(loss_remove_reduced.item()):.6f}"})
        
        if is_main and use_wandb and (iteration + 1) % 100 == 0:
            for diag_idx, diag_prompt in enumerate(diagnostic_prompts):
                # Get diagnostic embedding from cache
                diag_emb = hyper_cache.get(diag_prompt, accelerator.device)

                diag_time_steps = [0, hyper_train_steps // 2, hyper_train_steps]
                start_code_diag = torch.randn((1, 4, resolution // 8, resolution // 8), device=accelerator.device)
                imgs_per_prompt = []

                for h_step in diag_time_steps:
                    h_step_tensor = torch.tensor([h_step], device=accelerator.device)
                    base.hyper.set_context(diag_emb, h_step_tensor)
                    base.hyper.compute_and_cache_loras(diag_emb, h_step_tensor)

                    # Toggle LoRA for unconditional CFG pass inside CombinedCFGModel
                    combined_model = CombinedCFGModel(model=base).eval()
                    combined_sampler = DDIMSampler(model=combined_model)

                    imgs = generate_images(
                        sampler=combined_sampler,
                        model=combined_model,
                        prompt=diag_prompt,
                        device=accelerator.device,
                        steps=50,
                        guidance_scale=guidance_scale,
                        start_code=start_code_diag,
                    )
                    imgs_per_prompt.append(imgs)

                if len(imgs_per_prompt) > 0:
                    row_tensors = []
                    for imgs in imgs_per_prompt:
                        if imgs is None: continue
                        img = imgs[0].clamp(0, 1)
                        im_uint8 = (img * 255).round().to(torch.uint8).cpu()
                        row_tensors.append(im_uint8)
                    if len(row_tensors) > 0:
                        row = torch.cat(row_tensors, dim=2)
                        safe_key = diag_prompt.replace(" ", "_").replace(",", "")[:50]
                        wandb.log({f"diagnostic_{diag_idx}_{safe_key}": wandb.Image(to_pil_image(row), caption=f"{diag_prompt} | hyper steps: {diag_time_steps}")}, step=iteration)

            os.makedirs(output_dir, exist_ok=True)
            os.makedirs(final_save_path, exist_ok=True)

            # Save LoRA weights
            lora_state_dict = {}
            model_unwrapped = accelerator.unwrap_model(model)
            for name, param in model_unwrapped.model.diffusion_model.named_parameters():
                if param.requires_grad:
                    lora_state_dict[name] = param.detach().cpu().clone()

            lora_path = os.path.join(final_save_path, f"hyper_lora_{iteration}.pth")
            accelerator.save(lora_state_dict, lora_path)
            print(f"Model saved to: {lora_path}")

    accelerator.wait_for_everyone()
    if is_main:
        print(f"Final loss: {losses[-1]:.6f}")
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(final_save_path, exist_ok=True)
        model_unwrapped = accelerator.unwrap_model(model)
        lora_state_dict = {n: p.detach().cpu().clone() for n, p in model_unwrapped.model.diffusion_model.named_parameters() if p.requires_grad}
        lora_path = os.path.join(final_save_path, f"hyper_lora_final.pth")
        accelerator.save(lora_state_dict, lora_path)
        
        config_save = {
            "config_name": config_name,
            "concepts": concepts,
            "rank": rank_lora,
            "learning_rate_remove": learning_rate_remove,
            "learning_rate_retain": learning_rate_retain,
            "max_train_steps": max_train_steps,
            "celebrity_mode": celebrity_mode,
            "final_loss": losses[-1],
        }
        with open(os.path.join(final_save_path, "train_config.json"), "w") as f:
            json.dump(config_save, f, indent=2)

    # Save cache if modified during training
    if is_main and hyper_cache is not None and hyper_cache.dirty:
        print(f"[HyperCache] Saving updated cache...")
        hyper_cache.save(cache_path)

    if is_main and use_wandb:
        wandb.finish()

if __name__ == "__main__":
    main()
