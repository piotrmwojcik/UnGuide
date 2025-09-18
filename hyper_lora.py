import weakref
from typing import List

import torch
import torch.nn as nn


class HyperLora(nn.Module):

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        rank: int = 4,
        clip_size: int = 768,
        alpha: int = 16,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.rank = rank
        self.alpha = alpha

        if clip_size > 0:
            self.hypernetwork = nn.Linear(clip_size, (in_dim + out_dim) * rank)
        else:
            self.hypernetwork = None

        std_dev = 1 / (rank**0.5)
        self.A = nn.Parameter(torch.randn(in_dim, rank) * std_dev)
        self.B = nn.Parameter(torch.zeros(rank, out_dim))

    def forward(self, clip, x):
        if self.hypernetwork is not None and clip is not None:
            weights = self.hypernetwork(clip)
            A = (
                weights[: self.in_dim * self.rank]
                .contiguous()
                .view(self.in_dim, self.rank)
            )
            B = (
                weights[self.in_dim * self.rank :]
                .contiguous()
                .view(self.rank, self.out_dim)
            )
        else:
            A = self.A
            B = self.B

        return self.alpha * (x @ A @ B)


class HyperLoRALinear(nn.Module):

    def __init__(
        self,
        original_linear: nn.Linear,
        clip_size: int = 768,
        rank: int = 1,
        alpha: int = 16,
    ):
        super().__init__()
        self.original = original_linear
        self.hyper_lora = HyperLora(
            original_linear.in_features,
            original_linear.out_features,
            rank,
            clip_size,
            alpha,
        )
        self.parent_model = None

    def set_parent_model(self, model):
        self.parent_model = weakref.ref(model)

    def forward(self, x):
        # use the `()` for weakref
        parent = self.parent_model()
        clip_embedding = parent.current_conditioning
        if clip_embedding is None:
            print("WARNING: this shouldn't happen")
            return self.original(x)
        # Expected shape: (batch_size, seq_len, hidden_size)
        # e.g., (1, 77, 768)
        if clip_embedding.dim() == 3 and clip_embedding.shape[0] == 1:
            clip_embedding = clip_embedding[0]

        # Take the mean of the sequence of embeddings
        if clip_embedding.dim() == 2:
            clip_embedding = clip_embedding.mean(dim=0)

        return self.original(x) + self.hyper_lora(clip_embedding, x)


def inject_hyper_lora(
    module: nn.Module, target_modules: List[str], hyper_lora_factory, name: str = ""
):
    hyper_lora_layers = []

    for child_name, child in module.named_children():
        full_name = f"{name}.{child_name}" if name else child_name

        if isinstance(child, nn.Linear) and any(
            full_name.endswith(t) for t in target_modules
        ):
            device = next(child.parameters()).device
            hyper_lora_layer = hyper_lora_factory(child).to(device)
            setattr(module, child_name, hyper_lora_layer)
            hyper_lora_layers.append(hyper_lora_layer)
        else:
            child_layers = inject_hyper_lora(
                child, target_modules, hyper_lora_factory, full_name
            )
            hyper_lora_layers.extend(child_layers)

    return hyper_lora_layers


def inject_hyper_lora_nsfw(module, hyper_lora_factory, name=""):
    hyper_lora_layers = []

    for child_name, child in module.named_children():
        full_name = f"{name}.{child_name}" if name else child_name

        if (
            full_name.startswith("out.")
            or "attn2" in full_name
            or "time_embed" in full_name
        ):
            continue

        if isinstance(child, nn.Linear):
            print(f"Injecting HyperLoRA into: {full_name}")
            device = next(child.parameters()).device
            hyper_lora_layer = hyper_lora_factory(child).to(device)
            setattr(module, child_name, hyper_lora_layer)
            hyper_lora_layers.append(hyper_lora_layer)
        else:
            child_layers = inject_hyper_lora_nsfw(child, hyper_lora_factory, full_name)
            hyper_lora_layers.extend(child_layers)

    return hyper_lora_layers
