import weakref
from typing import List

import torch
from functools import partial
import torch.nn as nn


class HyperLora(nn.Module):

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        rank: int = 4,
        clip_size: int = 768,
        alpha: int = 16.0,
        use_scaling=True
    ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.rank = rank
        self.alpha = alpha

        self.layers = nn.Sequential(
            nn.Linear(clip_size, 100),
            nn.ReLU(),
        )

        self.left_head = nn.Linear(100, in_dim * rank)
        self.right_head = nn.Linear(100, out_dim * rank)
        self.in_dim = in_dim
        self.out_dim = out_dim

        self.use_scaling = use_scaling
        if self.use_scaling:
            self.alpha = nn.Parameter(torch.tensor(alpha))


    def forward(self, clip, x):
        x = self.layers(x)
        if self.use_scaling:
            x_L = self.alpha * self.left_head(x)
        else:
            x_L = self.left_head(x)
        x_R = self.right_head(x)
        return (
            x_L.view(-1, self.rank, self.in_dim),
            x_R.view(-1, self.rank, self.out_dim),
        )


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

        (x_L, x_R) = self.hyper_lora(clip_embedding, x)
        print(clip_embedding.shape, x_L.shape, x_R.shape)

        return self.original(x) + (clip_embedding @ x_L @ x_R)


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
