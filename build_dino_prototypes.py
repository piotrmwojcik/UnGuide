#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn.functional as F
import pandas as pd
from tqdm import tqdm

import timm
from timm.data import resolve_model_data_config, create_transform
from PIL import Image


def parse_args():
    ap = argparse.ArgumentParser("Build DINO class prototypes from generated images")
    ap.add_argument("--in_root",  type=str, required=True,
                    help="Input root (e.g., samples_cifar) with subfolders cifar100/cifar10/<class>/*.png")
    ap.add_argument("--out_root", type=str, required=True,
                    help="Output root to mirror structure and save prototypes")
    ap.add_argument("--model", type=str, default="vit_large_patch14_dinov2.lvd142m",
                    help="timm model name (DINO/DINOv2). Ex: vit_base_patch16_224.dino, vit_large_patch14_dinov2.lvd142m")
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--batch", type=int, default=32, help="Batch size for feature extraction")
    ap.add_argument("--robust", type=str, default="huber", choices=["mean", "trimmed", "huber"],
                    help="Robust aggregation method")
    ap.add_argument("--trim_frac", type=float, default=0.10,
                    help="Trim fraction for 'trimmed' mean (0.10 = drop 10% farthest)")
    ap.add_argument("--huber_delta", type=float, default=0.20,
                    help="Delta (in cosine distance) for Huber weighting")
    ap.add_argument("--max_images_per_class", type=int, default=0,
                    help="Optional cap per class (0 = use all)")
    return ap.parse_args()


@torch.no_grad()
def build_preprocess(model_name: str, img_size_override: int = None):
    """Create a torchvision transform matching the timm model."""
    dummy = timm.create_model(model_name, pretrained=True)
    cfg = resolve_model_data_config(dummy)
    tfm = create_transform(**cfg, is_training=False)
    # Optional override of input size if desired
    return tfm, cfg


@torch.no_grad()
def dino_cls_features(model, images: List[Image.Image], tfm, device: torch.device) -> torch.Tensor:
    """Encode a list of PIL images into L2-normalized CLS embeddings: [N, D]."""
    if len(images) == 0:
        return torch.empty(0, model.num_features, device=device)
    batch = torch.stack([tfm(img.convert("RGB")) for img in images]).to(device)  # [B,3,H,W]
    # timm ViTs: forward_features returns dict with tokens. Common key: x_norm_clstoken or 'cls_token'
    out = model.forward_features(batch)
    if isinstance(out, dict) and "x_norm_clstoken" in out:
        cls = out["x_norm_clstoken"]                                # [B, D]
    elif isinstance(out, dict) and "cls_token" in out:
        cls = out["cls_token"]                                      # [B, D]
    else:
        # fallback: assume forward returns [B,D]
        cls = out if isinstance(out, torch.Tensor) else out[0]
    return F.normalize(cls, dim=-1)


@torch.no_grad()
def extract_folder_features(model, tfm, folder: Path, batch: int, device, cap: int = 0) -> torch.Tensor:
    """Read all *.png (and *.jpg/jpeg) images in folder and return [N,D] normalized embeddings."""
    imgs_paths = sorted([*folder.glob("*.png"), *folder.glob("*.jpg"), *folder.glob("*.jpeg")])
    if cap > 0:
        imgs_paths = imgs_paths[:cap]
    feats = []
    for i in range(0, len(imgs_paths), batch):
        chunk = imgs_paths[i:i+batch]
        pil_list = [Image.open(p) for p in chunk]
        feats.append(dino_cls_features(model, pil_list, tfm, device))
        for im in pil_list:
            try:
                im.close()
            except Exception:
                pass
    if len(feats) == 0:
        return torch.empty(0, model.num_features, device=device)
    return torch.cat(feats, dim=0)  # [N,D]


def cosine_dist(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """1 - cosine(a,b) for matching shapes: [N,D]·[D] -> [N]. Assumes L2-normalized."""
    return 1.0 - (a @ b.unsqueeze(-1)).squeeze(-1)


@torch.no_grad()
def prototype_mean(feats: torch.Tensor) -> torch.Tensor:
    """Plain mean on the sphere: average then renormalize."""
    if feats.numel() == 0:
        return feats.new_zeros(feats.shape[-1])
    mu = feats.mean(dim=0, keepdim=True)
    return F.normalize(mu, dim=-1).squeeze(0)


@torch.no_grad()
def prototype_trimmed(feats: torch.Tensor, trim_frac: float = 0.10, iters: int = 2) -> torch.Tensor:
    """Trimmed mean: iteratively drop farthest trim_frac, then mean."""
    if feats.numel() == 0:
        return feats.new_zeros(feats.shape[-1])
    mu = prototype_mean(feats)
    for _ in range(max(1, iters)):
        d = cosine_dist(feats, mu)                 # [N]
        k = max(1, int((1.0 - trim_frac) * feats.size(0)))
        keep_idx = torch.topk(-d, k=k).indices     # keep smallest distances
        kept = feats.index_select(0, keep_idx)
        mu = prototype_mean(kept)
    return mu


@torch.no_grad()
def prototype_huber(feats: torch.Tensor, delta: float = 0.20, iters: int = 5) -> torch.Tensor:
    """
    Huber IRLS on cosine distance (robust to outliers).
    delta is a cosine-distance threshold (e.g., 0.2).
    """
    if feats.numel() == 0:
        return feats.new_zeros(feats.shape[-1])
    mu = prototype_mean(feats)
    for _ in range(max(1, iters)):
        r = cosine_dist(feats, mu).clamp(min=0)    # residuals in [0,2]
        # Huber weights (derivative of Huber loss): w = 1 if r<=δ else δ/r
        w = torch.where(r <= delta, torch.ones_like(r), delta / (r + 1e-8))  # [N]
        w = w / (w.sum() + 1e-8)
        # weighted mean on sphere
        mu = F.normalize((w.unsqueeze(1) * feats).sum(dim=0, keepdim=True), dim=-1).squeeze(0)
    return mu


def aggregate(feats: torch.Tensor, method: str, trim_frac: float, huber_delta: float) -> torch.Tensor:
    if method == "mean":
        return prototype_mean(feats)
    elif method == "trimmed":
        return prototype_trimmed(feats, trim_frac=trim_frac)
    else:
        return prototype_huber(feats, delta=huber_delta)


def main():
    args = parse_args()
    in_root = Path(args.in_root)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    model = timm.create_model(args.model, pretrained=True).to(device).eval()
    tfm, _ = build_preprocess(args.model)

    # Expect structure like: in_root/{cifar100,cifar10}/{class}/images.png
    subsets = []
    for name in ["cifar100", "cifar10"]:
        p = in_root / name
        if p.is_dir():
            subsets.append(name)

    rows = []
    total_classes = 0
    for subset in subsets:
        subset_in = in_root / subset
        subset_out = out_root / subset
        classes = sorted([d for d in subset_in.iterdir() if d.is_dir()])
        total_classes += len(classes)

    pbar = tqdm(total=total_classes, desc="Prototyping classes", unit="class")

    for subset in subsets:
        subset_in = in_root / subset
        subset_out = out_root / subset
        subset_out.mkdir(parents=True, exist_ok=True)

        classes = sorted([d for d in subset_in.iterdir() if d.is_dir()])
        for cdir in classes:
            cls_name = cdir.name
            feats = extract_folder_features(model, tfm, cdir, batch=args.batch, device=device,
                                            cap=args.max_images_per_class)
            N = feats.shape[0]
            if N == 0:
                proto = torch.zeros(model.num_features, device="cpu")
                avg_cos = float("nan")
                std_cos = float("nan")
            else:
                proto = aggregate(feats, args.robust, args.trim_frac, args.huber_delta)  # [D]
                # Dispersion stats
                cos = (feats @ proto)  # already unit-norm -> cosine
                avg_cos = float(cos.mean().item())
                std_cos = float(cos.std(unbiased=False).item())

            # Save mirror structure
            out_dir = subset_out / cls_name
            out_dir.mkdir(parents=True, exist_ok=True)
            torch.save(proto.cpu(), out_dir / "prototype.pt")
            with open(out_dir / "stats.json", "w") as f:
                json.dump({
                    "subset": subset,
                    "class": cls_name.replace("_", " "),
                    "images_used": int(N),
                    "robust": args.robust,
                    "trim_frac": args.trim_frac,
                    "huber_delta": args.huber_delta,
                    "avg_cosine_to_proto": avg_cos,
                    "std_cosine_to_proto": std_cos,
                }, f, indent=2)

            rows.append({
                "subset": subset,
                "class": cls_name,
                "images_used": N,
                "robust": args.robust,
                "avg_cosine_to_proto": avg_cos,
                "std_cosine_to_proto": std_cos,
                "proto_path": str((out_dir / "prototype.pt").resolve()),
            })

            pbar.update(1)

    pbar.close()
    # Global index
    if rows:
        df = pd.DataFrame(rows)
        df.to_csv(out_root / "index.csv", index=False)
        print(f"Wrote index: {out_root / 'index.csv'}")
    print(f"Done. Prototypes saved under: {out_root}")


if __name__ == "__main__":
    main()
