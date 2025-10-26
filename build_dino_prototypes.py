#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import List

import torch
import torch.nn.functional as F
import pandas as pd
from tqdm import tqdm

import timm
from timm.data import resolve_model_data_config, create_transform
from PIL import Image


def parse_args():
    ap = argparse.ArgumentParser("Build DINO class prototypes (simple mean)")
    ap.add_argument("--in_root",  type=str, required=True,
                    help="Input root (e.g., samples_cifar) with subfolders cifar100/cifar10/<class>/*.png")
    ap.add_argument("--out_root", type=str, required=True,
                    help="Output root to mirror structure and save prototypes")
    ap.add_argument("--model", type=str, default="vit_large_patch14_dinov2.lvd142m",
                    help="timm model name (e.g., vit_base_patch16_224.dino, vit_large_patch14_dinov2.lvd142m)")
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--batch", type=int, default=32, help="Batch size for feature extraction")
    ap.add_argument("--max_images_per_class", type=int, default=0,
                    help="Optional cap per class (0 = use all)")
    return ap.parse_args()


@torch.no_grad()
def build_preprocess(model_name: str):
    """Create a torchvision transform matching the timm model."""
    dummy = timm.create_model(model_name, pretrained=True)
    cfg = resolve_model_data_config(dummy)
    tfm = create_transform(**cfg, is_training=False)
    return tfm, cfg


@torch.no_grad()
def dino_cls_features(model, images: List[Image.Image], tfm, device: torch.device) -> torch.Tensor:
    """Encode a list of PIL images into L2-normalized CLS embeddings: [N, D]."""
    if len(images) == 0:
        return torch.empty(0, model.num_features, device=device)
    batch = torch.stack([tfm(img.convert("RGB")) for img in images]).to(device)  # [B,3,H,W]
    out = model.forward_features(batch)
    if isinstance(out, dict) and "x_norm_clstoken" in out:
        cls = out["x_norm_clstoken"]                                # [B, D]
    elif isinstance(out, dict) and "cls_token" in out:
        cls = out["cls_token"]                                      # [B, D]
    else:
        cls = out if isinstance(out, torch.Tensor) else out[0]
    return F.normalize(cls, dim=-1)


@torch.no_grad()
def extract_folder_features(model, tfm, folder: Path, batch: int, device, cap: int = 0) -> torch.Tensor:
    """Read all *.png/jpg/jpeg images in folder and return [N,D] normalized embeddings."""
    imgs_paths = sorted([*folder.glob("*.png"), *folder.glob("*.jpg"), *folder.glob("*.jpeg")])
    if cap > 0:
        imgs_paths = imgs_paths[:cap]
    feats = []
    for i in range(0, len(imgs_paths), batch):
        chunk = imgs_paths[i:i+batch]
        pil_list = [Image.open(p) for p in chunk]
        feats.append(dino_cls_features(model, pil_list, tfm, device))
        for im in pil_list:
            try: im.close()
            except Exception: pass
    if len(feats) == 0:
        return torch.empty(0, model.num_features, device=device)
    return torch.cat(feats, dim=0)  # [N,D]


@torch.no_grad()
def prototype_mean(feats: torch.Tensor) -> torch.Tensor:
    """Plain mean on the sphere: average then renormalize."""
    if feats.numel() == 0:
        return feats.new_zeros(feats.shape[-1])
    mu = feats.mean(dim=0, keepdim=True)
    return F.normalize(mu, dim=-1).squeeze(0)


def main():
    args = parse_args()
    in_root = Path(args.in_root)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    model = timm.create_model(args.model, pretrained=True).to(device).eval()
    tfm, _ = build_preprocess(args.model)

    # Expect: in_root/{cifar100,cifar10}/{class}/*.png
    subsets = [name for name in ("cifar100", "cifar10") if (in_root / name).is_dir()]

    rows = []
    total_classes = sum(
        len([d for d in (in_root / s).iterdir() if d.is_dir()])
        for s in subsets
    )
    pbar = tqdm(total=total_classes, desc="Prototyping classes", unit="class")

    for subset in subsets:
        subset_in = in_root / subset
        subset_out = out_root / subset
        subset_out.mkdir(parents=True, exist_ok=True)

        classes = sorted([d for d in subset_in.iterdir() if d.is_dir()])
        for cdir in classes:
            cls_name = cdir.name
            feats = extract_folder_features(
                model, tfm, cdir, batch=args.batch, device=device,
                cap=args.max_images_per_class
            )
            N = feats.shape[0]
            if N == 0:
                proto = torch.zeros(model.num_features, device="cpu")
                avg_cos = float("nan")
                std_cos = float("nan")
            else:
                proto = prototype_mean(feats)       # [D]
                cos = (feats @ proto)               # cosine similarity to prototype
                avg_cos = float(cos.mean().item())
                std_cos = float(cos.std(unbiased=False).item())

            out_dir = subset_out / cls_name
            out_dir.mkdir(parents=True, exist_ok=True)
            torch.save(proto.cpu(), out_dir / "prototype.pt")
            with open(out_dir / "stats.json", "w") as f:
                json.dump({
                    "subset": subset,
                    "class": cls_name.replace("_", " "),
                    "images_used": int(N),
                    "avg_cosine_to_proto": avg_cos,
                    "std_cosine_to_proto": std_cos,
                    "agg": "mean"
                }, f, indent=2)

            rows.append({
                "subset": subset,
                "class": cls_name,
                "images_used": N,
                "avg_cosine_to_proto": avg_cos,
                "std_cosine_to_proto": std_cos,
                "proto_path": str((out_dir / "prototype.pt").resolve()),
            })

            pbar.update(1)

    pbar.close()
    if rows:
        df = pd.DataFrame(rows)
        df.to_csv(out_root / "index.csv", index=False)
        print(f"Wrote index: {out_root / 'index.csv'}")
    print(f"Done. Prototypes saved under: {out_root}")


if __name__ == "__main__":
    main()
