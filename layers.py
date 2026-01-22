import os
import re
import torch
from diffusers import FluxPipeline

# Optional: if FLUX is gated, ensure auth is available
# export HF_TOKEN=...
# or run `huggingface-cli login`
token = os.environ.get("HF_TOKEN", None)

repo_id = "black-forest-labs/FLUX.1-dev"

pipe = FluxPipeline.from_pretrained(
    repo_id,
    torch_dtype=torch.bfloat16,
    token=token if token else True,   # uses stored token if logged in
)

transformer = pipe.transformer

# ---- 1) Print a compact list of attention module names ----
print("\n=== Attention modules (named_modules) ===")
attn_names = []
for name, mod in transformer.named_modules():
    # Adjust filter keywords as needed:
    if ".attn" in name or name.endswith(".attn") or "attention" in name.lower():
        # keep only leaf-ish / useful names (optional)
        attn_names.append((name, mod.__class__.__name__))

# Print first N
for n, (name, cls) in enumerate(attn_names[:200]):
    print(f"{n:03d}  {name:<80}  {cls}")
print(f"\nTotal attn-ish modules found: {len(attn_names)}")

# ---- 2) Print projection submodules under attention ----
print("\n=== Projection layers under attn (q/k/v/out variants) ===")
proj_pat = re.compile(r"(to_[qkv]|to_out|q_proj|k_proj|v_proj|out_proj|add_[qkv]_proj|add_out_proj)$")

proj_names = []
for name, mod in transformer.named_modules():
    if ".attn" in name and proj_pat.search(name):
        proj_names.append((name, mod.__class__.__name__))

for n, (name, cls) in enumerate(proj_names[:300]):
    print(f"{n:03d}  {name:<80}  {cls}")
print(f"\nTotal attn projection modules found: {len(proj_names)}")

# ---- 3) (Optional) Print parameter names that include your typical suffixes ----
print("\n=== Parameter names with attn + proj keywords (good for target_modules matching) ===")
param_pat = re.compile(r"(attn\..*(to_[qkv]|to_out|add_[qkv]_proj|q_proj|k_proj|v_proj|out_proj))")
hits = 0
for pname, p in transformer.named_parameters():
    if param_pat.search(pname):
        print(f"{pname:<100}  shape={tuple(p.shape)}")
        hits += 1
        if hits >= 200:
            break
print(f"\nPrinted {hits} matching parameters (cap=200).")
