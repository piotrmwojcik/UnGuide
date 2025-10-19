import torch
import torch.nn as nn

# --- your module ---
import math
import torch
import torch.nn as nn

class TimeFourier(nn.Module):
    def __init__(self, T=501, L=16):  # outputs 2L dims
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

class Model(nn.Module):
    def __init__(self, in_dim=8, T=151, L=16, out_dim=4):
        super().__init__()
        self.time_feat = TimeFourier(T=T, L=L)             # (B, 2L)
        self.proj = nn.Linear(in_dim + 2*L, out_dim)       # concat then linear
    def forward(self, x, t):
        t_feats = self.time_feat(t)                        # (B, 2L)
        x_cat = torch.cat([x, t_feats], dim=-1)            # (B, in_dim + 2L)
        return self.proj(x_cat)

# ---- demo ----
B, in_dim, out_dim, T, L = 3, 8, 4, 501, 16
x = torch.randn(B, in_dim)            # your regular input
t = torch.tensor([0, 250, 500])       # integer timesteps (B,)

time_fourier = TimeFourier(T=T, L=L)
feats = time_fourier(t)               # (B, 32)
print("time features shape:", feats.shape)

model = Model(in_dim=in_dim, T=T, L=L, out_dim=out_dim)
y = model(x, t)                       # (B, out_dim)
print("model output shape:", y.shape)
