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
from utils import get_models, print_trainable_parameters


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
    pretrained_model_path = config.get('pretrained_model_name_or_path', './models/sd-v1-4.ckpt')
    model_config_path = config.get('model_config', './configs/stable-diffusion/v1-inference.yaml')
    
    # Training settings
    ddim_steps = 50
    ddim_eta = 0.0
    start_guidance = 7.5
    negative_guidance = config.get('negative_guidance', 1.0)
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
    
    # Load models
    model_orig, sampler_orig, model, sampler_unused = get_models(
        model_config_path, pretrained_model_path, accelerator.device
    )
    
    # Freeze original model
    for p in model_orig.model.diffusion_model.parameters():
        p.requires_grad = False
    model_orig.eval()
    
    # Freeze trainable model backbone
    for p in model.model.diffusion_model.parameters():
        p.requires_grad = False
    
    # Setup HyperLoRA
    model.hyper = HypernetworkManager()
    
    clip_size = 768 if use_pooler else 512
    target_modules = ["attn2.to_k", "attn2.to_v"]
    
    hyper_lora_factory = partial(
        HyperLoRALinear,
        clip_size=clip_size,
        rank=rank,
        alpha=lora_alpha,
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
    
    optimizer = torch.optim.Adam(trainable_params, lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[300], gamma=0.5
    )
    
    # Prepare for distributed training
    model, optimizer = accelerator.prepare(model, optimizer)
    
    # Register HyperLoRA layers after prepare
    for layer_name, layer in hyper_lora_layers:
        layer.set_parent_model(accelerator.unwrap_model(model))
        accelerator.unwrap_model(model).hyper.add_hyperlora(layer_name, layer.hyper_lora)
    
    # Create sampler after prepare
    sampler = DDIMSampler(accelerator.unwrap_model(model))
    
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
        
        # Apply prompt separate steps for training and for hyper (the same as in train.yp)ation to retain prompts if enabled
        if augment_retain:
            print("Applying prompt augmentation to retain prompts")
            for prompt in base_prompts:
                augmented = prompt_augmentation(prompt, augment=True)
                retain_prompts.extend(augmented)
            print(f"Generated {len(retain_prompts)} retain prompts with augmentation")
        else:
            retain_prompts = base_prompts
            print(f"Using {len(retain_prompts)} retain prompts without augmentation")
        
        # Create embeddings for retain prompts
        for prompt in tqdm(retain_prompts, desc="Creating retain embeddings", disable=not is_main):
            inputs = encode(prompt)
            with torch.no_grad():
                if use_pooler:
                    emb = clip_text_encoder(inputs).pooler_output.detach()
                else:
                    emb = clip_text_encoder(inputs).last_hidden_state.detach()
            retain_embeddings.append(emb.squeeze())
    else:
        print("No retain CSV path provided or file not found. Skipping retain loss.")
    
    print(f"Mapping concepts: {mapping_concept[:2]}...")  # Show first 2
    print(f"Retain prompts: {len(retain_prompts)} prompts loaded")
    
    # Training loop
    criterion = torch.nn.MSELoss()
    losses = []
    
    quick_sampler = create_quick_sampler(
        accelerator.unwrap_model(model), sampler, resolution, ddim_steps, ddim_eta
    )
    
    pbar = tqdm(range(max_train_steps), disable=not accelerator.is_local_main_process)
    
    # Training weights for combining removal and retain losses
    removal_weight = config.get('removal_weight', 1.0)  # Weight for removal loss
    retain_weight = config.get('retain_weight', 0.001)  # Weight for retain loss
    
    print(f"Loss weights: removal={removal_weight:.3f}, retain={retain_weight:.3f}")
    
    for iteration in pbar:
        base = accelerator.unwrap_model(model)
        
        optimizer.zero_grad(set_to_none=True)
        
        # Random timestep
        t_enc = torch.randint(ddim_steps, (1,), device=accelerator.device)
        og_num = round((int(t_enc) / ddim_steps) * 1000)
        og_num_lim = round((int(t_enc + 1) / ddim_steps) * 1000)
        t_enc_ddpm = torch.randint(og_num, og_num_lim, (1,), device=accelerator.device)
        
        # Starting latent code
        start_code = torch.randn((1, 4, resolution // 8, resolution // 8), device=accelerator.device)
        
        loss_retain, loss_remove = None, None
        
        with accelerator.accumulate(model):
            # RETAIN LOSS: Ensure model doesn't forget other concepts
            if len(retain_embeddings) > 0:
                # Sample multiple retain concepts
                num_retain_samples = min(10, len(retain_embeddings))
                sampled_retain_embs = random.sample(retain_embeddings, num_retain_samples)
                
                # Batch process retain concepts
                batch_retain_embs = torch.stack(sampled_retain_embs, dim=0).to(accelerator.device)
                
                hyper = base.hyper
                batch_prompts = batch_retain_embs.repeat(hyper_train_steps // num_retain_samples, 1)
                B = batch_prompts.shape[0]
                perm = torch.randperm(B, device=batch_prompts.device)
                batch_prompts = batch_prompts[perm]
                
                # Compute LoRAs at t=0
                hyper.compute_and_cache_loras(
                    batch_prompts,
                    torch.zeros(B, device=accelerator.device)
                )
                tensors_flat_t0 = hyper.flatten_cached_from_cache()
                
                # Compute LoRAs at t=1, 2, 3, ... B
                t_ = (torch.arange(B, device=accelerator.device) % B) + 1
                hyper.compute_and_cache_loras(batch_prompts, t_)
                tensors_flat_t1 = hyper.flatten_cached_from_cache()
                
                # Loss: minimize change in LoRA weights across timesteps
                delta = tensors_flat_t1 - tensors_flat_t0
                loss_retain = retain_weight * delta.pow(2).mean()
                
                loss_retain_for_backward = loss_retain / accelerator.gradient_accumulation_steps
                accelerator.backward(loss_retain_for_backward)
            else:
                loss_retain = torch.tensor(0.0, device=accelerator.device)
            
            # REMOVAL LOSS: Push target concepts towards mapping concepts
            # Select random target concept
            concept_idx = random.randint(0, len(target_embeddings) - 1)
            target_emb = target_embeddings[concept_idx]
            
            target_text = target_concepts[concept_idx]
            mapping_text = mapping_concept[concept_idx] if concept_idx < len(mapping_concept) else mapping_concept[0]
            
            # Apply prompt augmentation to target if enabled
            # When augmenting, apply the SAME augmentation to both target and mapping
            if augment_target:
                augmented_prompts = prompt_augmentation(target_text, augment=True)
                # Pick a random augmentation variation
                aug_idx = random.randint(0, len(augmented_prompts) - 1)
                target_text_augmented = augmented_prompts[aug_idx]
                
                # Apply the SAME augmentation variation to mapping
                augmented_mapping = prompt_augmentation(mapping_text, augment=True)
                mapping_text_augmented = augmented_mapping[aug_idx]
            else:
                target_text_augmented = target_text
                mapping_text_augmented = mapping_text
            
            # Get text conditioning for Stable Diffusion
            emb_p = base.get_learned_conditioning([target_text_augmented])  # target prompt (positive)
            emb_n = base.get_learned_conditioning([target_text_augmented])  # target prompt (negative, to be erased)
            emb_m = base.get_learned_conditioning([mapping_text_augmented])  # mapping prompt (what target should map to)
            
            # Random timestep for HyperLoRA context
            rtimestep = int(torch.randint(0, hyper_train_steps - 1, (1,), device=accelerator.device))
            base.hyper.set_context(target_emb, torch.tensor([rtimestep], device=accelerator.device))
            
            _, current_timestep = base.hyper.get_context()
            base.hyper.compute_and_cache_loras(target_emb, current_timestep)
            
            with torch.no_grad():
                # Generate latent using target prompt
                z = quick_sampler(emb_p, start_guidance, start_code, int(t_enc))
                # Get noise predictions from original model
                e_m = model_orig.apply_model(z, t_enc_ddpm, emb_m)  # mapping (reference) concept
                e_p = model_orig.apply_model(z, t_enc_ddpm, emb_p)  # target prompt
            
            # Prediction from modified model (with HyperLoRA)
            _, current_timestep = base.hyper.get_context()
            base.hyper.compute_and_cache_loras(target_emb, current_timestep)
            base.hyper.retain_grad_for_cached_lora()
            e_n = base.apply_model(z, t_enc_ddpm, emb_n)
            
            # Loss: push modified output away from target, towards mapping concept
            e_m.requires_grad_(False)
            e_p.requires_grad_(False)
            target = e_m - (negative_guidance * (e_p - e_m))
            loss_aux = criterion(e_n, target)

            accelerator.backward(loss_aux / accelerator.gradient_accumulation_steps, retain_graph=True)

            # --- use cached LoRA grads instead of live-tensor grads ---
            grads_flat_t = base.hyper.flatten_cached_grads_from_cache()
            if grads_flat_t is None:
                raise RuntimeError(
                    "No gradients found in cached LoRA tensors. Ensure cache is built with graph intact and retain_grad() was called.")
            # Target step: Δθ ≈ -lr * g_t  (keep target detached)
            grads_flat_t = (-1.0 * internal_lr) * grads_flat_t.detach()
            
            _, current_timestep = base.hyper.get_context()
            base.hyper.set_context(target_emb, current_timestep)
            base.hyper.compute_and_cache_loras(target_emb, current_timestep)
            tensors_flat_t = base.hyper.flatten_cached_from_cache()
            
            base.hyper.set_context(target_emb, current_timestep + 1)
            base.hyper.compute_and_cache_loras(target_emb, current_timestep + 1)
            tensors_flat_t1 = base.hyper.flatten_cached_from_cache()
            
            # Match the SGD step: (θ_{t+1} - θ_t) ≈ -lr * g_t
            delta_live = tensors_flat_t1 - tensors_flat_t
            loss_remove = criterion(delta_live, grads_flat_t)
            loss_for_backward = loss_remove / accelerator.gradient_accumulation_steps
            loss_remove_log = loss_remove.clone().detach()
            loss_retain_log = loss_retain.clone().detach()
            
            accelerator.backward(loss_for_backward, retain=True)
            
            # Optimizer step
            if accelerator.sync_gradients:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
        
        # Combined total loss for logging
        loss = loss_remove + loss_retain
        
        # Gather loss across devices
        with torch.no_grad():
            loss_reduced = accelerator.gather(loss.detach()).mean()
            loss_retain_reduced = accelerator.gather(loss_remove_log).mean()
            loss_remove_reduced = accelerator.gather(loss_retain_log).mean()
        
        loss_value = float(loss_reduced.item())
        losses.append(loss_value)
        
        if is_main and use_wandb:
            wandb.log({
                "loss": loss_value,
                "loss_retain": float(loss_retain_reduced.item()),
                "loss_remove": float(loss_remove_reduced.item()),
            }, step=iteration)
        
        if is_main:
            pbar.set_postfix({
                "loss": f"{loss_value:.6f}",
                "remove": f"{float(loss_remove_reduced.item()):.6f}",
                "retain": f"{float(loss_retain_reduced.item()):.6f}"
            })
        
        # Generate sample images periodically
        if is_main and use_wandb and iteration % 20 == 0:
            # Generate images for diagnostic prompts from config
            for diag_idx, diag_prompt in enumerate(diagnostic_prompts):
                # Encode the diagnostic prompt
                inputs_diag = encode(diag_prompt)
                with torch.no_grad():
                    if use_pooler:
                        diag_emb = clip_text_encoder(inputs_diag).pooler_output.detach()
                    else:
                        diag_emb = clip_text_encoder(inputs_diag).last_hidden_state.detach()
                
                base.hyper.set_context(diag_emb, torch.tensor([hyper_train_steps], device=accelerator.device))
                base.hyper.compute_and_cache_loras(diag_emb, torch.tensor([hyper_train_steps], device=accelerator.device))
                
                # Use CombinedCFGModel: conditional uses model (with LoRA), unconditional uses model_orig
                combined_model = CombinedCFGModel(cond_model=base, uncond_model=model_orig).eval()
                combined_sampler = DDIMSampler(model=combined_model)
                
                start_code = torch.randn((1, 4, resolution // 8, resolution // 8), device=accelerator.device)
                
                imgs = generate_images(
                    sampler=combined_sampler,
                    model=combined_model,
                    prompt=diag_prompt,
                    device=accelerator.device,
                    steps=50,
                    guidance_scale=start_guidance,
                    start_code=start_code,
                )
                
                if imgs is not None:
                    im0 = (imgs[0].clamp(0, 1) * 255).round().to(torch.uint8).cpu()
                    # Clean prompt for wandb key (remove spaces and special chars)
                    safe_key = diag_prompt.replace(" ", "_").replace(",", "")[:50]
                    wandb.log({f"diagnostic_{diag_idx}_{safe_key}": wandb.Image(to_pil_image(im0), caption=diag_prompt)}, step=iteration)
    
    # Save model
    accelerator.wait_for_everyone()
    if is_main:
        print("\nTraining completed!")
        print(f"Final loss: {losses[-1]:.6f}")
        print(f"Average loss: {sum(losses)/len(losses):.6f}")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(final_save_path, exist_ok=True)
        
        # Save LoRA weights
        lora_state_dict = {}
        model_unwrapped = accelerator.unwrap_model(model)
        for name, param in model_unwrapped.model.diffusion_model.named_parameters():
            if param.requires_grad:
                lora_state_dict[name] = param.detach().cpu().clone()
        
        lora_path = os.path.join(final_save_path, "hyper_lora.pth")
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
