import weakref
from typing import List

import torch
import math
import os
import torch.nn.functional as F
from functools import partial
import torch.nn as nn


class TimeFourier(nn.Module):

    def __init__(self, T=151, L=16):  # outputs 2L dims
        super().__init__()
        # freq_k = (2π/T) * 2^k  for k=0..L-1
        k = torch.linspace(0, L - 1, L, dtype=torch.float32)
        freqs = (2.0 * math.pi / T) * torch.pow(torch.tensor(2.0), k)
        # make it follow .to(device) / .half() calls automatically
        self.register_buffer("freqs", freqs)  # shape: (L,)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # t: (B,) int/long
        t = t.to(dtype=torch.float32).unsqueeze(-1)  # (B,1)
        w = self.freqs.to(dtype=t.dtype)             # (L,)
        angles = t * w                               # (B,L)
        return torch.cat([angles.cos(), angles.sin()], dim=-1)  # (B, 2L)


class HyperLora(nn.Module):

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        rank: int = 4,
        clip_size: int = 768,
        alpha_init: int = 16.0,
        time_embedd: int = 32,
        use_scaling=True
    ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.rank = rank
        self._dbg_tag = f"{self.__class__.__name__}@{id(self):x}"
        self._dbg_calls = 0   # to avoid spamming
        ## it should (?) be shared
        #self.layers = nn.Sequential(
        #    nn.Linear(clip_size, 100),
        #    nn.ReLU(),
        #)
        std_dev = 1 / (rank ** 0.5)
        self.register_buffer(
            "xL_const_flat", torch.rand(1, in_dim * rank) * std_dev, persistent=False
        )
        self.register_buffer(
            "xR_const_flat", torch.zeros(1, out_dim * rank), persistent=False
        )
        self.register_buffer(
            "alpha_b", torch.tensor(alpha_init, dtype=torch.float32), persistent=False
        )
        self.left_head = nn.Sequential(
            nn.Linear(clip_size + time_embedd, 100),
            nn.ReLU(inplace=True),
            nn.Linear(100, in_dim * rank),
        )
        self.right_head = nn.Sequential(
            nn.Linear(clip_size + time_embedd, 100),
            nn.ReLU(inplace=True),
            nn.Linear(100, out_dim * rank),
        )
        self.time_feat = TimeFourier()
        self.in_dim = in_dim
        self.out_dim = out_dim

        self.use_scaling = use_scaling
        if self.use_scaling:
            self.alpha = nn.Parameter(torch.tensor(alpha_init))

    def forward_linear_L(self, emb, t):
        return self.xL_const_flat + t / 150 * self.left_head(emb)

    def forward_linear_R(self, emb, t):
        return self.xR_const_flat + t / 150 * self.right_head(emb)

    def forward_alpha(self, t):
        return self.alpha_b + t / 150 * self.alpha

    def forward(self, x, clip, t):
        B = clip.shape[0]
        emb = clip
        #emb = self.layers(clip)
        t_feats = torch.full((B,), t, dtype=x.dtype, device=x.device)
        t_feats = self.time_feat(t_feats).to(x.device)
        emb = torch.cat([emb, t_feats], dim=-1)
        if self.use_scaling:
            x_L = self.forward_alpha(t) * self.forward_linear_L(emb, t)
        else:
            x_L = self.forward_linear_L(emb, t)
        x_R = self.forward_linear_R(emb, t)
        x_L = x_L.view(-1, self.in_dim, self.rank)
        x_R = x_R.view(-1, self.rank, self.out_dim)

        if x_L.requires_grad:
            x_L.retain_grad()
        if x_R.requires_grad:
            x_R.retain_grad()
        # stash references so you can read .grad later
        self._last_x_L = x_L
        self._last_x_R = x_R

        return (x @ x_L) @ x_R


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
        self.time_feat = TimeFourier(T=151, L=16)
        self.parent_model = None

    def set_parent_model(self, model):
        self.parent_model = weakref.ref(model)

    def forward(self, x):
        # use the `()` for weakref
        parent = self.parent_model()
        assert parent.time_step is not None
        #assert parent.target_prompt is not None

        if parent.current_conditioning is None:
            print("WARNING: this shouldn't happen")
            return self.original(x)
        else:
            clip_embedding = parent.current_conditioning
        # Expected shape: (batch_size, seq_len, hidden_size)
        # e.g., (1, 77, 768)
        #if clip_embedding.dim() == 3 and clip_embedding.shape[0] == 1:
        #    clip_embedding = clip_embedding[0]

        # Take the mean of the sequence of embeddings
        #if clip_embedding.dim() == 2:
        #    clip_embedding = clip_embedding.mean(dim=0)
        #print(self.original(x).shape, self.hyper_lora(x, clip_embedding, parent.time_step).shape)
        # if parent.time_step == 0:
        #     # Print once
        #     if not hasattr(self, "_printed_original_trainables"):
        #         total = 0
        #         print("[original] Trainable parameters:")
        #         for name, p in self.original.named_parameters():
        #             if p.requires_grad:
        #                 print(f"  - {name} shape={tuple(p.shape)}  numel={p.numel():,}")
        #                 total += p.numel()
        #         print(f"[original] Total trainable params: {total:,}")
        #         self._printed_original_trainables = True
        #
        #     return self.original(x)
        return self.original(x) + self.hyper_lora(x, clip_embedding, parent.time_step)


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
