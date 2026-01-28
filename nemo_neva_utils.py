"""Utility functions for NeMo NeVA (LLaVA) embeddings."""

import torch
import torch.nn.functional as F
from typing import List, Tuple, Union


NEVA_MODEL_NAME = "llava-hf/llava-1.5-7b-hf"
NEVA_EMBED_DIM = 4096  # LLaMA-7B hidden size


def load_neva_model(
    device: Union[str, torch.device] = "cuda",
    dtype: torch.dtype = torch.float16,
) -> Tuple:
    """
    Load LLaVA-1.5-7B model for text embedding extraction.

    Downloads automatically to ~/.cache/huggingface/hub/ on first use (~14GB).

    Returns:
        (model, tokenizer) tuple
    """
    from transformers import LlavaForConditionalGeneration, AutoTokenizer

    model = LlavaForConditionalGeneration.from_pretrained(
        NEVA_MODEL_NAME,
        torch_dtype=dtype,
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    model.eval()

    # Use slow tokenizer to avoid compatibility issues with fast tokenizer
    tokenizer = AutoTokenizer.from_pretrained(NEVA_MODEL_NAME, use_fast=False)

    return model, tokenizer


def compute_neva_embed(
    prompts: Union[str, List[str]],
    model,
    tokenizer,
    device: Union[str, torch.device] = "cuda",
    batch_size: int = 8,
    max_length: int = 512,
) -> torch.Tensor:
    """
    Compute LLaVA text embeddings for prompts.

    Extracts the last hidden state from the language model backbone.

    Args:
        prompts: Single prompt string or list of prompts
        model: LLaVA model from load_neva_model
        tokenizer: Tokenizer from load_neva_model
        device: Device for computation
        batch_size: Batch size for encoding
        max_length: Maximum sequence length

    Returns:
        Tensor of shape [N, 4096] with normalized embeddings
    """
    if isinstance(prompts, str):
        prompts = [prompts]

    all_embeddings = []

    with torch.no_grad():
        for i in range(0, len(prompts), batch_size):
            batch = prompts[i:i + batch_size]

            # Tokenize text only (no image input)
            inputs = tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
            )

            # Move to device
            inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

            # Forward pass through the language model to get hidden states
            outputs = model.language_model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                output_hidden_states=True,
                return_dict=True,
            )

            # Get the last hidden state
            hidden_states = outputs.hidden_states[-1]  # [B, seq_len, 4096]

            # Use mean pooling over sequence (excluding padding)
            attention_mask = inputs["attention_mask"].unsqueeze(-1)  # [B, seq_len, 1]
            masked_hidden = hidden_states * attention_mask
            embeddings = masked_hidden.sum(dim=1) / attention_mask.sum(dim=1).clamp(min=1e-9)

            # Convert to float32 and normalize
            embeddings = embeddings.to(dtype=torch.float32)
            embeddings = F.normalize(embeddings, p=2, dim=-1)
            all_embeddings.append(embeddings)

    result = torch.cat(all_embeddings, dim=0)
    return result


def compute_neva_embed_single(
    prompt: str,
    model,
    tokenizer,
    device: Union[str, torch.device] = "cuda",
) -> torch.Tensor:
    """
    Compute NeVA embedding for a single prompt.

    Returns:
        Tensor of shape [1, 4096]
    """
    return compute_neva_embed([prompt], model, tokenizer, device, batch_size=1)
