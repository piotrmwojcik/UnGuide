#!/usr/bin/env python3
"""
Compute FID between two image directories using torchmetrics' FrechetInceptionDistance.

Example:
  python fid_torchmetrics.py --dir1 /path/to/real --dir2 /path/to/fake --batch-size 64 --device cuda

Notes:
- Expects image files under each directory (optionally including subfolders if --recursive).
- Uses uint8 images in [0, 255] as required by torchmetrics FID.
"""

import argparse
import os
from pathlib import Path
from typing import List, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import functional as TF
from torchmetrics.image.fid import FrechetInceptionDistance
from tqdm import tqdm


IMG_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}


def list_images(root: Path, recursive: bool) -> List[Path]:
    if recursive:
        files = [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMG_EXTS]
    else:
        files = [p for p in root.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS]
    files.sort()  # deterministic order
    return files


class ImageFolderDataset(Dataset):
    def __init__(self, paths: List[Path], resize: int | None = None):
        self.paths = paths
        self.resize = resize

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        p = self.paths[idx]
        with Image.open(p) as im:
            im = im.convert("RGB")
            if self.resize is not None:
                # Torchmetrics FID uses InceptionV3 internally; any consistent resize is ok.
                # If you want "standard-ish" behavior, resize both dirs to same size.
                im = im.resize((self.resize, self.resize), resample=Image.BICUBIC)

            x = TF.to_tensor(im)  # float32 in [0,1], shape [3,H,W]
            x = (x * 255.0).clamp(0, 255).to(torch.uint8)  # uint8 in [0,255]
            return x


def compute_fid_dir_to_dir(
    dir1: str,
    dir2: str,
    device: str = "cuda",
    batch_size: int = 64,
    num_workers: int = 4,
    feature: int = 2048,
    resize: int | None = None,
    recursive: bool = False,
    limit: int | None = None,
) -> float:
    d1 = Path(dir1)
    d2 = Path(dir2)

    if not d1.exists():
        raise FileNotFoundError(f"--dir1 not found: {d1}")
    if not d2.exists():
        raise FileNotFoundError(f"--dir2 not found: {d2}")

    paths1 = list_images(d1, recursive=recursive)
    paths2 = list_images(d2, recursive=recursive)

    if limit is not None:
        paths1 = paths1[:limit]
        paths2 = paths2[:limit]

    if len(paths1) == 0:
        raise ValueError(f"No images found in --dir1: {d1}")
    if len(paths2) == 0:
        raise ValueError(f"No images found in --dir2: {d2}")

    print(f"Found {len(paths1)} images in dir1: {d1}")
    print(f"Found {len(paths2)} images in dir2: {d2}")

    # FID is typically computed on equal-sized sets; torchmetrics can handle unequal,
    # but for comparability it’s common to match sizes by truncation.
    n = min(len(paths1), len(paths2))
    if len(paths1) != len(paths2):
        print(f"Warning: unequal counts; truncating both to {n} images for comparability.")
        paths1 = paths1[:n]
        paths2 = paths2[:n]

    ds1 = ImageFolderDataset(paths1, resize=resize)
    ds2 = ImageFolderDataset(paths2, resize=resize)

    dl1 = DataLoader(ds1, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    dl2 = DataLoader(ds2, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    fid_metric = FrechetInceptionDistance(feature=feature).to(device)

    # Update "real"
    for batch in tqdm(dl1, desc="FID dir1 (real)", unit="batch"):
        fid_metric.update(batch.to(device, non_blocking=True), real=True)

    # Update "fake"
    for batch in tqdm(dl2, desc="FID dir2 (fake)", unit="batch"):
        fid_metric.update(batch.to(device, non_blocking=True), real=False)

    score = fid_metric.compute().item()
    return score


def main(args):
    score = compute_fid_dir_to_dir(
        dir1=args.dir1,
        dir2=args.dir2,
        device=args.device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        feature=args.feature,
        resize=args.resize,
        recursive=args.recursive,
        limit=args.limit,
    )
    print(f"FID score: {score}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute FID score between two directories (torchmetrics).")
    parser.add_argument("--dir1", type=str, required=True, help="Path to the first directory (real)")
    parser.add_argument("--dir2", type=str, required=True, help="Path to the second directory (fake)")
    parser.add_argument("--device", type=str, default="cuda", help="Device, e.g. cuda, cuda:0, cpu")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for FID updates")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader workers")
    parser.add_argument(
        "--feature",
        type=int,
        default=64,
        help="Inception feature layer size for torchmetrics FID (commonly 64, 192, 768, 2048)",
    )
    parser.add_argument(
        "--resize",
        type=int,
        default=None,
        help="Optional resize to NxN before FID (applied to both dirs). If omitted, uses original sizes.",
    )
    parser.add_argument("--recursive", action="store_true", help="Scan directories recursively")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional cap on number of images from each directory (after sorting).",
    )

    args = parser.parse_args()
    main(args)
