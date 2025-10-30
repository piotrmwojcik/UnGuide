import weakref
from typing import List, Dict, Tuple, Optional

import torch
import math
import os
import torch.nn.functional as F
from functools import partial
import torch.nn as nn


class HypernetworkManager(nn.Module):
    def __init__(self):
        super().__init__()
        self.hyper_layers = nn.ModuleList()
        self.layer_name_to_idx = {}
        self.lora_weights_cache = {}
        self.current_context = {'clip_emb': None, 'timestep': None}
        self.auto_mode = True

    def add_hyperlora(self, name: str, hyper_lora):
        idx = len(self.hyper_layers)
        self.hyper_layers.append(hyper_lora)
        self.layer_name_to_idx[name] = idx

    def set_context(self, clip_emb, timestep):
        self.current_context['clip_emb'] = clip_emb
        self.current_context['timestep'] = timestep

    def get_context(self):
        return self.current_context['clip_emb'], self.current_context['timestep']

    def compute_and_cache_loras(self, clip_emb, timestep):
        self.lora_weights_cache.clear()
        for name, idx in self.layer_name_to_idx.items():
            hyper = self.hyper_layers[idx]
            x_L, x_R = hyper.get_lora_matrices(clip_emb,  timestep)
            self.lora_weights_cache[name] = (x_L, x_R)

    def get_cached_lora(self, layer_name):
        return self.lora_weights_cache.get(layer_name, None)


class TimeFourier(nn.Module):

    def __init__(self, T=151, L=16):
        super().__init__()
        k = torch.linspace(0, L - 1, L, dtype=torch.float32)
        freqs = (2.0 * math.pi / T) * torch.pow(torch.tensor(2.0), k)
        self.register_buffer("freqs", freqs)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t = t.to(dtype=torch.float32).unsqueeze(-1)
        w = self.freqs.to(dtype=t.dtype)
        angles = t * w
        return torch.cat([angles.cos(), angles.sin()], dim=-1)


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

        self.use_scaling = use_scaling
        if self.use_scaling:
            self.alpha = nn.Parameter(torch.tensor(alpha_init))

    def forward_linear_L(self, emb, t):
        return self.xL_const_flat + t / 150 * self.left_head(emb)

    def forward_linear_R(self, emb, t):
        return self.xR_const_flat + t / 150 * self.right_head(emb)

    def forward_alpha(self, t):
        return self.alpha_b + t / 150 * self.alpha

    def get_lora_matrices(self, clip, t):
        if isinstance(t, torch.Tensor):
            t = int(t.item())

        B = clip.shape[0]
        emb = clip
        t_feats = torch.full((B,), t, dtype=clip.dtype, device=clip.device)
        t_feats = self.time_feat(t_feats).to(clip.device)
        emb = torch.cat([emb, t_feats], dim=-1)

        if self.use_scaling:
            x_L = self.forward_alpha(t) * self.forward_linear_L(emb, t)
        else:
            x_L = self.forward_linear_L(emb, t)
        x_R = self.forward_linear_R(emb, t)

        x_L = x_L.view(-1, self.in_dim, self.rank)
        x_R = x_R.view(-1, self.rank, self.out_dim)

        return x_L, x_R

    def forward(self, x, clip, t):
        x_L, x_R = self.get_lora_matrices(clip, t)

        if x_L.requires_grad:
            x_L.retain_grad()
        if x_R.requires_grad:
            x_R.retain_grad()

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
        layer_name: str = None,
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
        self.layer_name = layer_name

    def set_parent_model(self, model):
        self.parent_model = weakref.ref(model)

    def forward(self, x):
        parent = self.parent_model()

        if hasattr(parent, 'hyper') and parent.hyper is not None:
            if parent.hyper.auto_mode:
                clip_embedding, timestep = parent.hyper.get_context()
                if clip_embedding is None:
                    print("WARNING: clip_embedding is None in auto mode")
                    return self.original(x)
                return self.original(x) + self.hyper_lora(x, clip_embedding, timestep)
            else:
                lora_weights = parent.hyper.get_cached_lora(self.layer_name)
                if lora_weights is None:
                    return self.original(x)
                x_L, x_R = lora_weights

                batch_size = x.shape[0]
                if x_L.shape[0] == 1 and batch_size > 1:
                    x_L = x_L.expand(batch_size, -1, -1)
                    x_R = x_R.expand(batch_size, -1, -1)

                return self.original(x) + (x @ x_L) @ x_R
        else:
            if not hasattr(parent, 'current_conditioning'):
                print("WARNING: parent model has neither 'hyper' nor 'current_conditioning'")
                return self.original(x)

            clip_embedding = parent.current_conditioning
            timestep = getattr(parent, 'time_step', None)

            if clip_embedding is None or timestep is None:
                return self.original(x)

            return self.original(x) + self.hyper_lora(x, clip_embedding, timestep)


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
            hyper_lora_layer.layer_name = full_name
            setattr(module, child_name, hyper_lora_layer)
            hyper_lora_layers.append((full_name, hyper_lora_layer))
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
            hyper_lora_layer.layer_name = full_name
            setattr(module, child_name, hyper_lora_layer)
            hyper_lora_layers.append((full_name, hyper_lora_layer))
        else:
            child_layers = inject_hyper_lora_nsfw(child, hyper_lora_factory, full_name)
            hyper_lora_layers.extend(child_layers)

    return hyper_lora_layers
