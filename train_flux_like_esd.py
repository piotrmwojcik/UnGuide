import os
import torch
import sys
import random
import argparse
import copy
import numpy as np
import gc
import json
import dotenv
from tqdm.auto import tqdm
from safetensors.torch import save_file
from functools import partial

# Load environment variables
dotenv.load_dotenv()

from diffusers import FluxPipeline, AutoencoderKL, FlowMatchEulerDiscreteScheduler
from diffusers.models import FluxTransformer2DModel
from diffusers.utils import make_image_grid
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration, set_seed as hf_set_seed
from transformers import CLIPTokenizer, T5TokenizerFast, PretrainedConfig
import yaml
import pandas as pd
from diffusers.pipelines.flux.pipeline_flux import calculate_shift, retrieve_timesteps

from hyper_lora import HyperLoRALinear, HypernetworkManager, inject_hyper_lora

# --- Utilities ---

def prompt_augmentation(content, augment=True):
    """Generate augmented prompts for a given concept."""
    if augment:
        prompts = [
            f"A photo of {content}", f"A photograph of {content}", f"A picture of {content}",
            f"A close-up photo of {content}", f"A snapshot of {content}", f"A photograph showcasing {content}",
            f"An illustration of {content}", f"A digital rendering of {content}", f"A visual representation of {content}",
            f"A graphic of {content}", f"A shot of {content}", f"A photo of {content}",
            f"A black and white image of {content}", f"A depiction in portrait form of {content}",
            f"A scene depicting {content} during a public gathering", f"{content} captured in an image",
            f"A depiction created with oil paints capturing {content}", f"An image of {content}",
            f"A drawing capturing the essence of {content}", f"An official photograph featuring {content}",
            f"A detailed sketch of {content}", f"{content} during sunset/sunrise",
            f"{content} in a detailed portrait", f"An official photo of {content}",
            f"Historic photo of {content}", f"Detailed portrait of {content}",
            f"A painting of {content}", f"HD picture of {content}",
            f"Magazine cover capturing {content}", f"Painting-like image of {content}",
            f"Hand-drawn art of {content}", f"An oil portrait of {content}",
            f"{content} in a sketch painting",
        ]
        return prompts
    else:
        return [content]

def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    config_name = list(config.keys())[0]
    return config[config_name], config_name

def parse_args():
    parser = argparse.ArgumentParser(description="HyperLoRA Training for Flux (Merged)")
    parser.add_argument('--config', type=str, required=True, help="Path to YAML configuration file")
    # NEW FLAG: Controls aggressive VRAM optimization
    parser.add_argument('--low_memory', action='store_true', help="Offload/Remove text encoders to save VRAM (recommended for <80GB VRAM)")
    return parser.parse_args()

def import_model_class_from_model_name_or_path(pretrained_model_name_or_path: str, revision: str, subfolder: str = "text_encoder"):
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

# --- Flux Model Loader ---

def load_flux_models(basemodel_id="black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16, device='cuda:0'):
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(basemodel_id, subfolder="scheduler")
    
    text_encoder_cls_one = import_model_class_from_model_name_or_path(basemodel_id, None)
    text_encoder_cls_two = import_model_class_from_model_name_or_path(basemodel_id, None, subfolder="text_encoder_2")
    
    text_encoder = text_encoder_cls_one.from_pretrained(basemodel_id, subfolder="text_encoder", torch_dtype=torch_dtype)
    text_encoder_2 = text_encoder_cls_two.from_pretrained(basemodel_id, subfolder="text_encoder_2", torch_dtype=torch_dtype)
    
    tokenizer = CLIPTokenizer.from_pretrained(basemodel_id, subfolder="tokenizer")
    tokenizer_2 = T5TokenizerFast.from_pretrained(basemodel_id, subfolder="tokenizer_2")
    
    vae = AutoencoderKL.from_pretrained(basemodel_id, subfolder="vae", torch_dtype=torch.float32)
    transformer = FluxTransformer2DModel.from_pretrained(basemodel_id, subfolder="transformer", torch_dtype=torch_dtype)

    pipe = FluxPipeline(
        transformer=transformer,
        scheduler=scheduler,
        vae=vae,
        text_encoder=text_encoder,
        text_encoder_2=text_encoder_2,
        tokenizer=tokenizer,
        tokenizer_2=tokenizer_2,
    )
    
    return pipe, transformer, scheduler

# --- Core Training Logic ---

@torch.no_grad()
def get_noisy_latents(pipe, prompt_embeds, pooled_prompt_embeds, text_ids, num_inference_steps, run_till_timestep, batch_size, height, width, generator, device, dtype):
    num_channels_latents = pipe.transformer.config.in_channels // 4
    h_latent = height // 8
    w_latent = width // 8
    
    # 1. Generate random latents
    latents = torch.randn((batch_size, num_channels_latents, h_latent, w_latent), generator=generator, device=device, dtype=dtype)
    
    # 2. Pack latents (Flux specific packing)
    latents = pipe._pack_latents(latents, batch_size, num_channels_latents, h_latent, w_latent)
    
    # 3. Timesteps
    sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
    image_seq_len = latents.shape[1]
    mu = calculate_shift(
        image_seq_len, pipe.scheduler.config.base_image_seq_len, pipe.scheduler.config.max_image_seq_len,
        pipe.scheduler.config.base_shift, pipe.scheduler.config.max_shift,
    )
    timesteps, num_inference_steps = retrieve_timesteps(
        pipe.scheduler, num_inference_steps, device, None, sigmas, mu=mu,
    )
    
    # 4. Prepare Image IDs (Use halved dimensions for packed latents)
    latent_image_ids = pipe._prepare_latent_image_ids(batch_size, h_latent // 2, w_latent // 2, device, dtype)
    
    prompt_embeds = prompt_embeds.to(dtype)
    pooled_prompt_embeds = pooled_prompt_embeds.to(dtype)
    text_ids = text_ids.to(dtype)
    
    # Handle text_ids batch dimension
    if text_ids.ndim == 3:
        text_ids = text_ids[0]

    # Guidance vector
    guidance_vec = torch.full((batch_size,), 3.5, device=device, dtype=dtype)
    
    current_timestep = None
    for i, t in enumerate(timesteps):
        if i >= run_till_timestep:
            current_timestep = t
            break
        vec_t = t.expand(latents.shape[0]).to(dtype)
        
        noise_pred = pipe.transformer(
            hidden_states=latents, timestep=vec_t / 1000, guidance=guidance_vec, 
            pooled_projections=pooled_prompt_embeds, encoder_hidden_states=prompt_embeds, 
            txt_ids=text_ids, img_ids=latent_image_ids, return_dict=False,
        )[0]
        
        latents = pipe.scheduler.step(noise_pred, t, latents, return_dict=False)[0]
        current_timestep = t

    return latents, latent_image_ids, current_timestep

# --- Main Execution ---

def main():
    args = parse_args()
    config, config_name = load_config(args.config)
    
    print(f"=== Training with config: {config_name} ===")
    if args.low_memory:
        print(">>> LOW MEMORY MODE ENABLED (Pre-computing & Removing Encoders) <<<")
    
    # Extract params
    pretrained_model_name_or_path = config.get('pretrained_model_name_or_path', "black-forest-labs/FLUX.1-dev")
    max_train_steps = config.get('max_train_steps', 120)
    hyper_train_steps = config.get('hyper_train_steps', 500)
    rank = config.get('rank', 1)
    lora_alpha = config.get('lora_alpha', 8)
    internal_size = config.get('internal_size', 100)
    seed = config.get('seed', 2024)
    resolution = config.get('resolution', 512)
    gradient_accumulation_steps = config.get('gradient_accumulation_steps', 1)
    output_dir = config.get('output_dir', './output')
    final_save_path = config.get('final_save_path', './saved_model/LoRA_fusion_model')
    
    learning_rate_remove = config.get('learning_rate_remove', 1e-5)
    learning_rate_retain = config.get('learning_rate_retain', 1e-5)
    weight_remove = config.get('remove_weight', 1.0)
    weight_retain = config.get('retain_weight', 1.0)
    
    negative_guidance = config.get('negative_guidance', 2.0)
    internal_lr = config.get('internal_lr', 1e-4)
    use_orig_concat = config.get('use_orig_concat', False)
    
    augment_target = config.get('augment_target', True)
    augment_retain = config.get('augment_retain', False)
    concepts = config.get('concepts', [])
    mapping_concept = config.get('mapping_concept', [])
    retain_csv_path = config.get('retain_csv_path', None)
    
    # Diagnostic
    diagnostic_prompts = config.get('diagnostic_prompts', [])
    if not diagnostic_prompts:
        diagnostic_prompts = [f"a photo of {concepts[0]}" if concepts else "a photo of a person"]

    # Scheduler Params
    gamma = config.get('gamma', 0.9)
    step_size = config.get('step_size', 300)
    drop_lr_on_plateau = config.get('drop_lr_on_plateau', False)
    plateau_factor = config.get('plateau_factor', 0.1)
    plateau_patience_remove = config.get('plateau_patience_remove', 10)
    plateau_patience_retain = config.get('plateau_patience_retain', 10)

    # W&B check
    try:
        import wandb
        WANDB_AVAILABLE = hasattr(wandb, 'init')
    except ImportError:
        wandb = None
        WANDB_AVAILABLE = False

    # Prep Data
    all_augmented_prompts = []
    target_concepts = concepts if isinstance(concepts, list) else [concepts]
    if augment_target:
        for concept in target_concepts:
            all_augmented_prompts.extend(prompt_augmentation(concept, augment=True))
    else:
        all_augmented_prompts = target_concepts.copy()

    all_augmented_mapping = []
    if len(mapping_concept) > 0:
        mapping_per_target = []
        for i, concept in enumerate(target_concepts):
            if i < len(mapping_concept):
                mapping_per_target.append(mapping_concept[i])
            else:
                mapping_per_target.append(mapping_concept[0])
        if augment_target:
            for mapping_text in mapping_per_target:
                all_augmented_mapping.extend(prompt_augmentation(mapping_text, augment=True))
        else:
            all_augmented_mapping = mapping_per_target.copy()
    else:
        all_augmented_mapping = []

    retain_prompts = []
    if retain_csv_path and os.path.exists(retain_csv_path):
        df = pd.read_csv(retain_csv_path)
        base_prompts = df['prompt'].dropna().tolist()
        if augment_retain:
            for prompt in base_prompts:
                if prompt.startswith("A photo of the "):
                    prompt = prompt[len("A photo of the "):]
                retain_prompts.extend(prompt_augmentation(prompt, augment=True))
        else:
            retain_prompts = base_prompts
    else:
        retain_prompts = [config.get('retain_concept', 'a photo of a person')]

    # Accelerator
    project_config = ProjectConfiguration(project_dir=output_dir, logging_dir="logs")
    accelerator = Accelerator(
        mixed_precision=config.get('mixed_precision', 'bf16'),
        gradient_accumulation_steps=gradient_accumulation_steps,
        project_config=project_config,
        log_with=config.get('report_to', 'wandb') if WANDB_AVAILABLE else None
    )
    
    is_main = accelerator.is_main_process
    
    hf_set_seed(seed)

    # Models
    weight_dtype = torch.bfloat16 if accelerator.mixed_precision == "bf16" else torch.float32
    pipe, transformer, scheduler = load_flux_models(pretrained_model_name_or_path, torch_dtype=weight_dtype, device=accelerator.device)
    
    # Freeze
    pipe.text_encoder.requires_grad_(False)
    pipe.text_encoder_2.requires_grad_(False)
    pipe.vae.requires_grad_(False)
    transformer.requires_grad_(False)
    
    if args.low_memory:
        print("Enabling Gradient Checkpointing to save VRAM...")
        transformer.enable_gradient_checkpointing()

    # HyperLoRA
    transformer.hyper = HypernetworkManager()
    hyper_lora_factory = partial(
        HyperLoRALinear, clip_size=768, rank=rank, alpha=lora_alpha,
        train_steps=hyper_train_steps, use_orig_concat=use_orig_concat,
        dtype=torch.float32, internal_size=internal_size
    )
    target_modules = [
        # Image Stream
        "attn.to_k",
        "attn.to_q",
        
        # Text Stream
        "attn.add_k_proj",
        "attn.add_q_proj",
    ]
    hyper_lora_layers = inject_hyper_lora(transformer, target_modules, hyper_lora_factory)
    
    for layer_name, layer in hyper_lora_layers:
        layer.set_parent_model(transformer)

    # Optimizers
    trainable_params = [p for p in transformer.parameters() if p.requires_grad]
    optimizer_remove = torch.optim.Adam(trainable_params, lr=learning_rate_remove)
    optimizer_retain = torch.optim.Adam(trainable_params, lr=learning_rate_retain)

    # Schedulers
    if drop_lr_on_plateau:
        scheduler_remove = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer_remove, mode='min', factor=plateau_factor, patience=plateau_patience_remove, verbose=True
        )
        scheduler_retain = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer_retain, mode='min', factor=plateau_factor, patience=plateau_patience_retain, verbose=True
        )
    else:
        milestones = step_size if isinstance(step_size, list) else [step_size]
        scheduler_remove = torch.optim.lr_scheduler.MultiStepLR(optimizer_remove, milestones=milestones, gamma=gamma)
        scheduler_retain = torch.optim.lr_scheduler.MultiStepLR(optimizer_retain, milestones=milestones, gamma=gamma)

    # Prepare
    transformer, optimizer_remove, optimizer_retain = accelerator.prepare(transformer, optimizer_remove, optimizer_retain)
    unwrapped_model = accelerator.unwrap_model(transformer)
    for layer_name, layer in hyper_lora_layers:
        layer.set_parent_model(unwrapped_model)
        unwrapped_model.hyper.add_hyperlora(layer_name, layer.hyper_lora)

    # =========================================================================
    # PRE-COMPUTATION SECTION
    # =========================================================================
    print("Computing Prompt Embeddings (Target/Mapping)...")
    # Move encoders to GPU for computation
    pipe.text_encoder.to(accelerator.device)
    pipe.text_encoder_2.to(accelerator.device)

    # 1. Target & Mapping Embeddings (For the current concept pair)
    # We sample ONE pair for the whole run to keep it simple, or you can loop later.
    with torch.no_grad():
        target_text = all_augmented_prompts[random.randint(0, len(all_augmented_prompts)-1)]
        mapping_text = all_augmented_mapping[random.randint(0, len(all_augmented_mapping)-1)] if all_augmented_mapping else ""
        
        prompt_embeds_all, pooled_prompt_embeds_all, text_ids = pipe.encode_prompt(
            [target_text, mapping_text], prompt_2=[target_text, mapping_text], max_sequence_length=512
        )
        emb_target, emb_map = prompt_embeds_all.chunk(2)
        pooled_target, pooled_map = pooled_prompt_embeds_all.chunk(2)
        
        if text_ids.ndim == 3 and text_ids.shape[0] == 2:
            text_ids_target, text_ids_map = text_ids.chunk(2)
        else:
            text_ids_target = text_ids
            text_ids_map = text_ids
        
        hyper_emb_target = pooled_target[0:1].detach()

    # 2. Process ALL Retain Prompts (COCO)
    # We only need 'pooled_prompt_embeds' (768-dim) for HyperNetwork regularization.
    print(f"Pre-computing pooled embeddings for ALL {len(retain_prompts)} retain prompts...")
    all_retain_pooled_list = []
    encode_batch_size = 128 
    
    for i in tqdm(range(0, len(retain_prompts), encode_batch_size), desc="Encoding Retain Set"):
        batch_prompts = retain_prompts[i : i + encode_batch_size]
        with torch.no_grad():
            # encode_prompt returns (prompt_embeds, pooled_prompt_embeds, text_ids)
            # We ignore prompt_embeds (T5) to save massive RAM
            _, batch_pooled, _ = pipe.encode_prompt(
                batch_prompts, 
                prompt_2=batch_prompts, 
                max_sequence_length=77 # Standard for CLIP
            )
            # Store in CPU list
            all_retain_pooled_list.append(batch_pooled.cpu())

    # Concatenate into one large tensor [N, 768]
    if len(all_retain_pooled_list) > 0:
        all_retain_pooled_tensor = torch.cat(all_retain_pooled_list, dim=0)
        print(f"Retain Embeddings Shape: {all_retain_pooled_tensor.shape} (Size: {all_retain_pooled_tensor.element_size() * all_retain_pooled_tensor.nelement() / 1024**2:.2f} MB)")
    else:
        print("Warning: No retain prompts found. Using target pooled embedding as fallback.")
        all_retain_pooled_tensor = hyper_emb_target.cpu()

    # 3. Pre-compute Diagnostic Embeddings
    print("Pre-computing Diagnostic Embeddings (to avoid loading T5 later)...")
    diagnostic_cache = {}
    for diag_prompt in diagnostic_prompts:
        with torch.no_grad():
            pe, ppe, tids = pipe.encode_prompt(
                prompt=diag_prompt, prompt_2=diag_prompt, max_sequence_length=512
            )
            # Store in CPU dict
            diagnostic_cache[diag_prompt] = {
                "prompt_embeds": pe.cpu(),
                "pooled_prompt_embeds": ppe.cpu(),
                "text_ids": tids.cpu()
            }

    # 4. Remove Encoders if Low Memory
    if args.low_memory:
        print("Removing Text Encoders from memory completely to save VRAM...")
        # Set to None to prevent usage and allow GC
        pipe.text_encoder = None
        pipe.text_encoder_2 = None
        gc.collect()
        torch.cuda.empty_cache()
    else:
        pipe.text_encoder.to("cpu")
        pipe.text_encoder_2.to("cpu")
        torch.cuda.empty_cache()

    # =========================================================================
    # TRAINING LOOP
    # =========================================================================

    if is_main and WANDB_AVAILABLE and config.get('report_to') == 'wandb':
        wandb.init(project="UnGuide", name=f"{config_name}_training", config=config)

    print("Starting Training...")
    progress_bar = tqdm(range(max_train_steps), disable=not is_main)
    loss_fn = torch.nn.MSELoss()
    losses = []
    
    batch_size = config.get('batch_size', 1)

    for step in progress_bar:
        # Zero Grads (Memory Efficient)
        optimizer_remove.zero_grad(set_to_none=True)
        optimizer_retain.zero_grad(set_to_none=True)
        
        with accelerator.accumulate(transformer):
            run_till = random.randint(0, config.get('num_inference_steps', 28) - 1)
            
            # Use pre-computed training embeddings (moved to GPU)
            txt_ids_input = text_ids_target.to(accelerator.device, dtype=weight_dtype)
            if txt_ids_input.ndim == 3: 
                txt_ids_input = txt_ids_input[0]

            with torch.no_grad():
                with unwrapped_model.hyper.no_lora():
                    latents_t, latent_ids, t_tensor = get_noisy_latents(
                        pipe, emb_target, pooled_target, txt_ids_input,
                        config.get('num_inference_steps', 28), run_till, batch_size, resolution, resolution,
                        None, accelerator.device, weight_dtype
                    )

            hyper_t_idx = random.randint(0, hyper_train_steps - 1)
            hyper_t_tensor = torch.tensor([hyper_t_idx], device=accelerator.device, dtype=weight_dtype)
            
            unwrapped_model.hyper.set_context(hyper_emb_target.to(dtype=weight_dtype), hyper_t_tensor)
            unwrapped_model.hyper.compute_and_cache_loras(hyper_emb_target.to(dtype=weight_dtype), hyper_t_tensor)
            # IMPORTANT: retain_grad to allow gradient flow from loss_aux to hypernetwork
            unwrapped_model.hyper.retain_grad_for_cached_lora()

            guidance_vec = torch.full((batch_size,), 3.0, device=accelerator.device, dtype=weight_dtype)
            t_input = t_tensor.expand(batch_size).to(dtype=weight_dtype) / 1000
            
            txt_ids_target_curr = text_ids_target.to(accelerator.device, dtype=weight_dtype)
            txt_ids_map_curr = text_ids_map.to(accelerator.device, dtype=weight_dtype)
            
            if txt_ids_target_curr.ndim == 3: txt_ids_target_curr = txt_ids_target_curr[0]
            if txt_ids_map_curr.ndim == 3: txt_ids_map_curr = txt_ids_map_curr[0]

            with torch.no_grad():
                with unwrapped_model.hyper.no_lora():
                    e_p_base = transformer(
                        hidden_states=latents_t, timestep=t_input, guidance=guidance_vec,
                        pooled_projections=pooled_target, encoder_hidden_states=emb_target,
                        txt_ids=txt_ids_target_curr, img_ids=latent_ids, return_dict=False
                    )[0]
                    current_txt_ids = txt_ids_map_curr if mapping_text else txt_ids_target_curr
                    e_0 = transformer(
                        hidden_states=latents_t, timestep=t_input, guidance=guidance_vec,
                        pooled_projections=pooled_map if mapping_text else pooled_target,
                        encoder_hidden_states=emb_map if mapping_text else emb_target,
                        txt_ids=current_txt_ids, img_ids=latent_ids, return_dict=False
                    )[0]

            e_n = transformer(
                hidden_states=latents_t, timestep=t_input, guidance=guidance_vec,
                pooled_projections=pooled_target, encoder_hidden_states=emb_target,
                txt_ids=txt_ids_target_curr, img_ids=latent_ids, return_dict=False
            )[0]
            
            target_signal = e_0 - negative_guidance * (e_p_base - e_0)
            loss_aux = loss_fn(e_n.float(), target_signal.float())
            
            # Backward Pass 1: Get gradients w.r.t LoRA weights
            accelerator.backward(loss_aux)
            
            grads_flat = unwrapped_model.hyper.flatten_cached_grads_from_cache()
            
            if grads_flat is not None:
                # Detach to stop gradient flow back to loss_aux
                target_delta = (-1.0 * internal_lr) * grads_flat.detach()
                
                # Clear accumulated gradients from loss_aux
                optimizer_remove.zero_grad(set_to_none=True)
                
                # Re-run HyperNet for Loss Remove (Fresh Graph)
                # t state
                unwrapped_model.hyper.set_context(hyper_emb_target.to(dtype=weight_dtype), hyper_t_tensor)
                unwrapped_model.hyper.compute_and_cache_loras(hyper_emb_target.to(dtype=weight_dtype), hyper_t_tensor)
                tensors_flat_t = unwrapped_model.hyper.flatten_cached_from_cache()
                
                # t+1 state
                t_next = hyper_t_tensor + 1
                unwrapped_model.hyper.set_context(hyper_emb_target.to(dtype=weight_dtype), t_next)
                unwrapped_model.hyper.compute_and_cache_loras(hyper_emb_target.to(dtype=weight_dtype), t_next)
                tensors_flat_t1 = unwrapped_model.hyper.flatten_cached_from_cache()
                
                delta_live = tensors_flat_t1 - tensors_flat_t
                loss_remove = weight_remove * loss_fn(delta_live, target_delta)
                
                # Backward Pass 2: Optimize HyperNet
                accelerator.backward(loss_remove)
                optimizer_remove.step()
            else:
                loss_remove = torch.tensor(0.0)

        # --- Retain Step (Sampled from ALL COCO) ---
        if len(retain_prompts) > 0:
            optimizer_retain.zero_grad(set_to_none=True)
            
            # Randomly sample indices from the pre-computed 30k+ tensor
            num_samples = 8 
            indices = torch.randint(0, len(all_retain_pooled_tensor), (num_samples,))
            
            # Fetch embeddings and move to GPU
            hyper_retain_emb = all_retain_pooled_tensor[indices].to(accelerator.device, dtype=weight_dtype)
            
            # Calculate at t=0
            unwrapped_model.hyper.compute_and_cache_loras(hyper_retain_emb, torch.zeros(num_samples, device=accelerator.device))
            retain_t0 = unwrapped_model.hyper.flatten_cached_from_cache()
            
            # Calculate at t=1
            unwrapped_model.hyper.compute_and_cache_loras(hyper_retain_emb, torch.ones(num_samples, device=accelerator.device))
            retain_t1 = unwrapped_model.hyper.flatten_cached_from_cache()
            
            # Minimize change
            loss_retain = weight_retain * (retain_t1 - retain_t0).pow(2).mean()
            accelerator.backward(loss_retain)
            optimizer_retain.step()
        else:
            loss_retain = torch.tensor(0.0)

        if drop_lr_on_plateau:
            scheduler_remove.step(loss_remove.detach())
            scheduler_retain.step(loss_retain.detach())
        else:
            scheduler_remove.step()
            scheduler_retain.step()

        losses.append(loss_remove.item() + loss_retain.item())
        if is_main and WANDB_AVAILABLE and config.get('report_to') == 'wandb':
            wandb.log({
                "loss_retain": float(loss_retain.item()),
                "loss_remove": float(loss_remove.item()),
                "lr_remove": optimizer_remove.param_groups[0]['lr'],
                "lr_retain": optimizer_retain.param_groups[0]['lr'],
            }, step=step)
        
        progress_bar.set_postfix(rem=f"{loss_remove.item():.2e}", ret=f"{loss_retain.item():.2e}")

        # Image Gen (Using Pre-computed Embeddings)
        if is_main and (step + 1) % 100 == 0:
            print("Generating diagnostic images...")
            
            # Clear state
            optimizer_remove.zero_grad(set_to_none=True)
            optimizer_retain.zero_grad(set_to_none=True)
            gc.collect()
            torch.cuda.empty_cache()

            # Ensure VAE is on GPU (usually small enough)
            pipe.vae.to(accelerator.device)

            for diag_prompt in diagnostic_prompts:
                # Retrieve pre-computed embeddings
                d_data = diagnostic_cache.get(diag_prompt)
                if not d_data: continue
                
                pe = d_data["prompt_embeds"].to(accelerator.device, dtype=weight_dtype)
                ppe = d_data["pooled_prompt_embeds"].to(accelerator.device, dtype=weight_dtype)
                # Text IDs are inside 'prompt_embeds' or handled by pipe if we pass embeds
                # Note: FluxPipeline normally handles text_ids if prompt_embeds is passed.
                # However, to be safe, we can manually pass what we cached if pipe supports it,
                # or rely on standard pipe behavior which computes ids if missing (but we killed encoders).
                
                # We need to rely on pipe using passed embeddings ONLY.
                # FluxPipeline's __call__ accepts prompt_embeds and pooled_prompt_embeds.
                
                with torch.no_grad():
                    img = pipe(
                        prompt_embeds=pe,
                        pooled_prompt_embeds=ppe,
                        height=512, width=512, num_inference_steps=28,
                        guidance_scale=3.5, generator=torch.Generator(device=accelerator.device).manual_seed(seed)
                    ).images[0]
                    
                    if WANDB_AVAILABLE and config.get('report_to') == 'wandb':
                        wandb.log({f"diag_{diag_prompt}": wandb.Image(img)}, step=step)
            
            # Cleanup VAE if super tight
            # pipe.vae.to("cpu") 
            gc.collect()
            torch.cuda.empty_cache()

        # Save Checkpoint
        if is_main and ((step + 1) % 100 == 0 or step == max_train_steps - 1):
            os.makedirs(final_save_path, exist_ok=True)
            lora_path = os.path.join(final_save_path, f"hyper_lora_{step}.pth")
            state_dict = {k: v.cpu() for k, v in unwrapped_model.named_parameters() if v.requires_grad}
            torch.save(state_dict, lora_path)
            print(f"Saved: {lora_path}")

    if is_main:
        config_save = {"config_name": config_name, "final_loss": losses[-1], "average_loss": sum(losses)/len(losses), **config}
        with open(os.path.join(final_save_path, "train_config.json"), "w") as f:
            json.dump(config_save, f, indent=2)
        if WANDB_AVAILABLE and config.get('report_to') == 'wandb':
            wandb.finish()

if __name__ == "__main__":
    main()