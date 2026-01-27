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
    parser.add_argument('--low_memory', action='store_true', help="Offload/Remove text encoders to save VRAM")
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
    
    latents = torch.randn((batch_size, num_channels_latents, h_latent, w_latent), generator=generator, device=device, dtype=dtype)
    latents = pipe._pack_latents(latents, batch_size, num_channels_latents, h_latent, w_latent)
    
    sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
    image_seq_len = latents.shape[1]
    mu = calculate_shift(
        image_seq_len, pipe.scheduler.config.base_image_seq_len, pipe.scheduler.config.max_image_seq_len,
        pipe.scheduler.config.base_shift, pipe.scheduler.config.max_shift,
    )
    timesteps, num_inference_steps = retrieve_timesteps(
        pipe.scheduler, num_inference_steps, device, None, sigmas, mu=mu,
    )
    
    latent_image_ids = pipe._prepare_latent_image_ids(batch_size, h_latent // 2, w_latent // 2, device, dtype)
    
    prompt_embeds = prompt_embeds.to(dtype)
    pooled_prompt_embeds = pooled_prompt_embeds.to(dtype)
    text_ids = text_ids.to(dtype)
    
    if text_ids.ndim == 3:
        text_ids = text_ids[0]

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
    
    diagnostic_prompts = config.get('diagnostic_prompts', [])
    if not diagnostic_prompts:
        diagnostic_prompts = [f"a photo of {concepts[0]}" if concepts else "a photo of a person"]

    gamma = config.get('gamma', 0.9)
    step_size = config.get('step_size', 300)
    drop_lr_on_plateau = config.get('drop_lr_on_plateau', False)
    plateau_factor = config.get('plateau_factor', 0.1)
    plateau_patience_remove = config.get('plateau_patience_remove', 10)
    plateau_patience_retain = config.get('plateau_patience_retain', 10)

    try:
        import wandb
        WANDB_AVAILABLE = hasattr(wandb, 'init')
    except ImportError:
        wandb = None
        WANDB_AVAILABLE = False

    # --- Data Preparation ---
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
            
        if len(all_augmented_mapping) != len(all_augmented_prompts):
             while len(all_augmented_mapping) < len(all_augmented_prompts):
                 all_augmented_mapping.extend(all_augmented_mapping[:len(all_augmented_prompts) - len(all_augmented_mapping)])
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

    project_config = ProjectConfiguration(project_dir=output_dir, logging_dir="logs")
    accelerator = Accelerator(
        mixed_precision=config.get('mixed_precision', 'bf16'),
        gradient_accumulation_steps=gradient_accumulation_steps,
        project_config=project_config,
        log_with=config.get('report_to', 'wandb') if WANDB_AVAILABLE else None
    )
    
    is_main = accelerator.is_main_process

    if is_main and WANDB_AVAILABLE and config.get('report_to') == 'wandb':
        wandb.init(project="UnGuide", name=f"{config_name}_training", config=config)

    hf_set_seed(seed)

    weight_dtype = torch.bfloat16 if accelerator.mixed_precision == "bf16" else torch.float32
    pipe, transformer, scheduler = load_flux_models(pretrained_model_name_or_path, torch_dtype=weight_dtype, device=accelerator.device)
    
    pipe.text_encoder.requires_grad_(False)
    pipe.text_encoder_2.requires_grad_(False)
    pipe.vae.requires_grad_(False)
    transformer.requires_grad_(False)
    
    if args.low_memory:
        print("Enabling VAE Slicing and Tiling for Low Memory...")
        pipe.vae.enable_slicing()
        pipe.vae.enable_tiling()
        print("Enabling Gradient Checkpointing for Transformer...")
        transformer.enable_gradient_checkpointing()

    transformer.hyper = HypernetworkManager()
    hyper_lora_factory = partial(
        HyperLoRALinear, clip_size=768, rank=rank, alpha=lora_alpha,
        train_steps=hyper_train_steps, use_orig_concat=use_orig_concat,
        dtype=torch.float32, internal_size=internal_size
    )
    
    target_modules = config.get('target_modules', ["attn.add_v_proj", "attn.to_v", "attn.to_out.0"])
    hyper_lora_layers = inject_hyper_lora(transformer, target_modules, hyper_lora_factory)

    for layer_name, layer in hyper_lora_layers:
        layer.set_parent_model(transformer)

    trainable_params = [p for p in transformer.parameters() if p.requires_grad]
    optimizer_remove = torch.optim.Adam(trainable_params, lr=learning_rate_remove)
    optimizer_retain = torch.optim.Adam(trainable_params, lr=learning_rate_retain)

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

    transformer, optimizer_remove, optimizer_retain = accelerator.prepare(transformer, optimizer_remove, optimizer_retain)
    unwrapped_model = accelerator.unwrap_model(transformer)
    for layer_name, layer in hyper_lora_layers:
        layer.set_parent_model(unwrapped_model)
        unwrapped_model.hyper.add_hyperlora(layer_name, layer.hyper_lora)

    # =========================================================================
    # PRE-COMPUTATION SECTION
    # =========================================================================
    print("Computing Prompt Embeddings (Target/Mapping/Retain)...")
    pipe.text_encoder.to(accelerator.device)
    pipe.text_encoder_2.to(accelerator.device)

    target_embeds_cache = []
    mapping_embeds_cache = []

    print(f"Pre-computing {len(all_augmented_prompts)} target/mapping pairs...")
    
    chunk_size = 16
    for i in range(0, len(all_augmented_prompts), chunk_size):
        batch_targets = all_augmented_prompts[i:i+chunk_size]
        batch_mappings = all_augmented_mapping[i:i+chunk_size] if all_augmented_mapping else [""] * len(batch_targets)
        
        for t_txt, m_txt in zip(batch_targets, batch_mappings):
            with torch.no_grad():
                pe, ppe, tids = pipe.encode_prompt(
                    [t_txt, m_txt], prompt_2=[t_txt, m_txt], max_sequence_length=512
                )
                emb_t, emb_m = pe.chunk(2)
                pool_t, pool_m = ppe.chunk(2)
                
                target_embeds_cache.append({
                    "prompt_embeds": emb_t.cpu(),
                    "pooled_prompt_embeds": pool_t.cpu(),
                    "pooled_ctx": pool_t[0:1].detach().cpu() 
                })
                
                if m_txt:
                    mapping_embeds_cache.append({
                        "prompt_embeds": emb_m.cpu(),
                        "pooled_prompt_embeds": pool_m.cpu(),
                    })
                else:
                    mapping_embeds_cache.append(None) 

    print(f"Pre-computing pooled embeddings for ALL {len(retain_prompts)} retain prompts...")
    all_retain_pooled_list = []
    encode_batch_size = 128 
    
    for i in tqdm(range(0, len(retain_prompts), encode_batch_size), desc="Encoding Retain Set"):
        batch_prompts = retain_prompts[i : i + encode_batch_size]
        with torch.no_grad():
            _, batch_pooled, _ = pipe.encode_prompt(
                batch_prompts, 
                prompt_2=batch_prompts, 
                max_sequence_length=77 
            )
            all_retain_pooled_list.append(batch_pooled.cpu())

    if len(all_retain_pooled_list) > 0:
        all_retain_pooled_tensor = torch.cat(all_retain_pooled_list, dim=0)
    else:
        all_retain_pooled_tensor = target_embeds_cache[0]["pooled_ctx"].cpu()

    print("Pre-computing Diagnostic Embeddings...")
    diagnostic_cache = {}
    for diag_prompt in diagnostic_prompts:
        with torch.no_grad():
            pe, ppe, tids = pipe.encode_prompt(
                prompt=diag_prompt, prompt_2=diag_prompt, max_sequence_length=512
            )
            diagnostic_cache[diag_prompt] = {
                "prompt_embeds": pe.cpu(),
                "pooled_prompt_embeds": ppe.cpu(),
            }

    if args.low_memory:
        print("Removing Text Encoders from memory completely...")
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

    print("Starting Training...")
    progress_bar = tqdm(range(max_train_steps), disable=not is_main)
    loss_fn = torch.nn.MSELoss()
    losses = []
    
    batch_size = config.get('batch_size', 1)

    for step in progress_bar:
        # Zero Grads
        optimizer_remove.zero_grad(set_to_none=True)
        optimizer_retain.zero_grad(set_to_none=True)
        
        with accelerator.accumulate(transformer):
            rank_idx = accelerator.process_index
            world_size = accelerator.num_processes
            
            valid_indices = list(range(rank_idx, len(target_embeds_cache), world_size))
            if not valid_indices:
                concept_idx = rank_idx % len(target_embeds_cache)
            else:
                concept_idx = random.choice(valid_indices)

            t_data = target_embeds_cache[concept_idx]
            emb_target = t_data["prompt_embeds"].to(accelerator.device, dtype=weight_dtype)
            pooled_target = t_data["pooled_prompt_embeds"].to(accelerator.device, dtype=weight_dtype)
            hyper_emb_target = t_data["pooled_ctx"].to(accelerator.device, dtype=weight_dtype)
            
            # FIX: Manually create text_ids to match embedding length (Flux uses zeros for text)
            # This prevents 1536 vs 1280 mismatch if encode_prompt returns 256-length ids
            text_ids_target = torch.zeros(emb_target.shape[0], emb_target.shape[1], 3, device=accelerator.device, dtype=weight_dtype)

            if mapping_embeds_cache and mapping_embeds_cache[concept_idx] is not None:
                m_data = mapping_embeds_cache[concept_idx]
                emb_map = m_data["prompt_embeds"].to(accelerator.device, dtype=weight_dtype)
                pooled_map = m_data["pooled_prompt_embeds"].to(accelerator.device, dtype=weight_dtype)
                text_ids_map = torch.zeros(emb_map.shape[0], emb_map.shape[1], 3, device=accelerator.device, dtype=weight_dtype)
                has_mapping = True
            else:
                has_mapping = False
                emb_map, pooled_map, text_ids_map = None, None, None

            run_till = random.randint(0, config.get('num_inference_steps', 28) - 1)
            
            txt_ids_input = text_ids_target
            if txt_ids_input.ndim == 3: 
                txt_ids_input = txt_ids_input[0]

            # -------------------------------------------------------------
            # 1. Get Latents (Live)
            # -------------------------------------------------------------
            with torch.no_grad():
                with unwrapped_model.hyper.no_lora():
                    latents_t, latent_ids, t_tensor = get_noisy_latents(
                        pipe, emb_target, pooled_target, txt_ids_input,
                        config.get('num_inference_steps', 28), run_till, batch_size, resolution, resolution,
                        None, accelerator.device, weight_dtype
                    )

            # -------------------------------------------------------------
            # 2. HyperLoRA Setup (Exactly like simple_slow)
            # -------------------------------------------------------------
            hyper_t_idx = random.randint(0, hyper_train_steps - 1)
            hyper_t_tensor = torch.tensor([hyper_t_idx], device=accelerator.device, dtype=weight_dtype)
            
            # Initial Set Context
            unwrapped_model.hyper.set_context(hyper_emb_target, hyper_t_tensor)
            _, current_timestep = unwrapped_model.hyper.get_context()
            unwrapped_model.hyper.compute_and_cache_loras(hyper_emb_target, current_timestep)
            #unwrapped_model.hyper.retain_grad_for_cached_lora() # Not used here in simple_slow loop yet

            guidance_vec = torch.full((batch_size,), 3.0, device=accelerator.device, dtype=weight_dtype)
            t_input = t_tensor.expand(batch_size).to(dtype=weight_dtype) / 1000
            
            if text_ids_target.ndim == 3: text_ids_target = text_ids_target[0]
            if has_mapping and text_ids_map.ndim == 3: text_ids_map = text_ids_map[0]

            # -------------------------------------------------------------
            # 3. Predictions (No LoRA)
            # -------------------------------------------------------------
            with torch.no_grad():
                with unwrapped_model.hyper.no_lora():
                    # e_p (Positive/Target)
                    e_p_base = transformer(
                        hidden_states=latents_t, timestep=t_input, guidance=guidance_vec,
                        pooled_projections=pooled_target, encoder_hidden_states=emb_target,
                        txt_ids=text_ids_target, img_ids=latent_ids, return_dict=False
                    )[0]
                    
                    # e_0 (Mapping/Neutral)
                    if has_mapping:
                        e_0 = transformer(
                            hidden_states=latents_t, timestep=t_input, guidance=guidance_vec,
                            pooled_projections=pooled_map, encoder_hidden_states=emb_map,
                            txt_ids=text_ids_map, img_ids=latent_ids, return_dict=False
                        )[0]
                    else:
                        e_0 = e_p_base

            # -------------------------------------------------------------
            # 4. Predictions (With LoRA) & Context Reset
            # -------------------------------------------------------------
            # simple_slow re-sets context here before forward pass
            unwrapped_model.hyper.set_context(hyper_emb_target, current_timestep)
            _, current_timestep = unwrapped_model.hyper.get_context()
            unwrapped_model.hyper.compute_and_cache_loras(hyper_emb_target, current_timestep)
            unwrapped_model.hyper.retain_grad_for_cached_lora()

            e_n = transformer(
                hidden_states=latents_t, timestep=t_input, guidance=guidance_vec,
                pooled_projections=pooled_target, encoder_hidden_states=emb_target,
                txt_ids=text_ids_target, img_ids=latent_ids, return_dict=False
            )[0]
            
            # -------------------------------------------------------------
            # 5. Loss Aux (ESD)
            # -------------------------------------------------------------
            target_signal = (e_0 - negative_guidance * (e_p_base - e_0)).float()
            loss_aux = loss_fn(e_n.float(), target_signal)
            
            accelerator.backward(loss_aux)
            
            # -------------------------------------------------------------
            # 6. Loss Remove (Gradient Matching)
            # -------------------------------------------------------------
            grads_flat = unwrapped_model.hyper.flatten_cached_grads_from_cache()
            
            if grads_flat is not None:
                target_delta = (-1.0 * internal_lr) * grads_flat.detach()
                
                # Retrieve current context (like simple_slow)
                _, current_timestep = unwrapped_model.hyper.get_context()
                
                # State t
                unwrapped_model.hyper.set_context(hyper_emb_target, current_timestep)
                unwrapped_model.hyper.compute_and_cache_loras(hyper_emb_target, current_timestep)
                tensors_flat_t = unwrapped_model.hyper.flatten_cached_from_cache()

                # State t+1
                unwrapped_model.hyper.set_context(hyper_emb_target, current_timestep + 1)
                unwrapped_model.hyper.compute_and_cache_loras(hyper_emb_target, current_timestep + 1)
                tensors_flat_t1 = unwrapped_model.hyper.flatten_cached_from_cache()
                
                delta_live = tensors_flat_t1 - tensors_flat_t
                loss_remove = weight_remove * loss_fn(delta_live, target_delta)
                
                accelerator.backward(loss_remove)
                optimizer_remove.step()
            else:
                loss_remove = torch.tensor(0.0)

        # -------------------------------------------------------------
        # 7. Retain Step
        # -------------------------------------------------------------
        if len(retain_prompts) > 0:
            optimizer_retain.zero_grad(set_to_none=True)
            
            num_samples = 8 
            indices = torch.randint(0, len(all_retain_pooled_tensor), (num_samples,))
            hyper_retain_emb = all_retain_pooled_tensor[indices].to(accelerator.device, dtype=weight_dtype)
            
            # t = 0
            unwrapped_model.hyper.compute_and_cache_loras(hyper_retain_emb, torch.zeros(num_samples, device=accelerator.device))
            retain_t0 = unwrapped_model.hyper.flatten_cached_from_cache()
            
            # t = dynamic
            dtype_hyper = next(unwrapped_model.hyper.parameters()).dtype
            t_retain_dynamic = (torch.arange(num_samples, device=accelerator.device, dtype=dtype_hyper) % num_samples) + 1
            
            unwrapped_model.hyper.compute_and_cache_loras(hyper_retain_emb, t_retain_dynamic)
            retain_t1 = unwrapped_model.hyper.flatten_cached_from_cache()
            
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

        # Image Gen
        if is_main and (step + 1) % 100 == 0:
            print("Generating diagnostic images...")
            
            optimizer_remove.zero_grad(set_to_none=True)
            optimizer_retain.zero_grad(set_to_none=True)
            gc.collect()
            torch.cuda.empty_cache()

            if args.low_memory:
                transformer.to("cpu")
                torch.cuda.empty_cache()

            for diag_prompt in diagnostic_prompts:
                d_data = diagnostic_cache.get(diag_prompt)
                if not d_data: continue
                
                pe = d_data["prompt_embeds"].to(accelerator.device, dtype=weight_dtype)
                ppe = d_data["pooled_prompt_embeds"].to(accelerator.device, dtype=weight_dtype)
                
                # FIX: Create text_ids for diagnostic as well
                tids = torch.zeros(pe.shape[0], pe.shape[1], 3, device=accelerator.device, dtype=weight_dtype)

                transformer.to(accelerator.device)
                
                hyper_step_tensor = torch.tensor([hyper_train_steps - 1], device=accelerator.device, dtype=weight_dtype)
                model_for_gen = accelerator.unwrap_model(transformer)
                
                model_for_gen.hyper.set_context(ppe, hyper_step_tensor)
                model_for_gen.hyper.compute_and_cache_loras(ppe, hyper_step_tensor)
                
                with torch.no_grad():
                    latents = torch.randn((1, 16, 64, 64), device=accelerator.device, dtype=weight_dtype)
                    latents = pipe._pack_latents(latents, 1, 16, 64, 64)
                    
                    num_inference_steps = 28
                    sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
                    
                    image_seq_len = latents.shape[1]
                    mu = calculate_shift(
                        image_seq_len, pipe.scheduler.config.base_image_seq_len, pipe.scheduler.config.max_image_seq_len,
                        pipe.scheduler.config.base_shift, pipe.scheduler.config.max_shift,
                    )
                    
                    timesteps, _ = retrieve_timesteps(
                        pipe.scheduler, num_inference_steps, device=accelerator.device, 
                        timesteps=None, sigmas=sigmas, mu=mu
                    )
                    
                    latent_image_ids = pipe._prepare_latent_image_ids(1, 32, 32, accelerator.device, weight_dtype)
                    guidance = torch.tensor([3.5], device=accelerator.device, dtype=weight_dtype)
                    
                    for t in timesteps:
                        vec_t = t.expand(latents.shape[0]).to(dtype=weight_dtype)
                        noise_pred = transformer(
                            hidden_states=latents,
                            timestep=vec_t / 1000,
                            guidance=guidance,
                            pooled_projections=ppe,
                            encoder_hidden_states=pe,
                            txt_ids=tids,
                            img_ids=latent_image_ids,
                            return_dict=False
                        )[0]
                        latents = pipe.scheduler.step(noise_pred, t, latents, return_dict=False)[0]
                
                if args.low_memory:
                    transformer.to("cpu")
                    torch.cuda.empty_cache()
                
                pipe.vae.to(accelerator.device)
                
                with torch.no_grad():
                    scale_factor = getattr(pipe, 'vae_scale_factor', 8)
                    latents = pipe._unpack_latents(latents, 512, 512, scale_factor)
                    latents = (latents / pipe.vae.config.scaling_factor) + pipe.vae.config.shift_factor
                    
                    image = pipe.vae.decode(latents, return_dict=False)[0]
                    image = pipe.image_processor.postprocess(image, output_type="pil")[0]
                    
                    if WANDB_AVAILABLE and config.get('report_to') == 'wandb':
                        wandb.log({f"diag_{diag_prompt}": wandb.Image(image)}, step=step)
                
                if args.low_memory:
                    pipe.vae.to("cpu")
                    torch.cuda.empty_cache()

            transformer.to(accelerator.device)
            gc.collect()
            torch.cuda.empty_cache()

        if is_main and ((step + 1) % 100 == 0 or step == max_train_steps - 1):
            os.makedirs(final_save_path, exist_ok=True)
            lora_path = os.path.join(final_save_path, f"hyper_lora_{step}.pth")
            state_dict = {k: v.cpu() for k, v in unwrapped_model.named_parameters() if ".hyper_lora." in k}
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