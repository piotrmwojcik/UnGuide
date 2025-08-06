
import torch
import torch.nn as nn
from typing import List

class LoRALayer(nn.Module):
    """Low-Rank Adaptation layer implementation"""

    def __init__(self, in_dim: int, out_dim: int, rank: int = 4, alpha: int = 16):
        super().__init__()
        self.rank = rank
        self.alpha = alpha

        # Initialize LoRA matrices
        std_dev = 1 / (rank**0.5)
        self.A = nn.Parameter(torch.randn(in_dim, rank) * std_dev)
        self.B = nn.Parameter(torch.zeros(rank, out_dim))

    def forward(self, x):
        # Ensure tensors are on the same device
        A = self.A.to(x.device)
        B = self.B.to(x.device)
        return self.alpha * (x @ A @ B)


class LoRALinear(nn.Module):
    """Linear layer with LoRA adaptation"""

    def __init__(self, original_linear: nn.Linear, rank: int = 1, alpha: int = 16):
        super().__init__()
        self.original = original_linear
        self.lora = LoRALayer(
            original_linear.in_features, original_linear.out_features, rank, alpha
        )

    def forward(self, x):
        return self.original(x) + self.lora(x)


def inject_lora(
    module: nn.Module, target_modules: List[str], lora_factory, name: str = ""
):
    """Recursively inject LoRA layers into target modules"""
    for child_name, child in module.named_children():
        full_name = f"{name}.{child_name}" if name else child_name

        # Check if this is a target module for LoRA injection
        if isinstance(child, nn.Linear) and any(
            full_name.endswith(t) for t in target_modules
        ):
            # print(f"Injecting LoRA into: {full_name}")
            setattr(module, child_name, lora_factory(child))
        else:
            # Recursively process children
            inject_lora(child, target_modules, lora_factory, full_name)

def inject_lora_nsfw(module, name="", lora_factory=None):
    for child_name, child in module.named_children():
        full_name = f"{name}.{child_name}" if name else child_name
        if (
            full_name.startswith("out.") or
            "attn2" in full_name or
            "time_embed" in full_name
        ):
            continue
        if isinstance(child, nn.Linear):
            print(f"Injecting LoRA into: {full_name}")
            setattr(module, child_name, lora_factory(child))
        else:
            inject_lora_nsfw(child, full_name, lora_factory)