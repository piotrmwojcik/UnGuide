#!/usr/bin/env python3
"""
Compute FID between two image directories using StyleGAN2-ADA TorchScript Inception
(inception-2015-12-05.pt) and the classic FID formula.

Example:
  python fid_stylegan_inception.py --dir1 /path/real --dir2 /path/fake --device cuda --batch-size 64 --resize 256

Notes:
- This matches the detector used in many StyleGAN2-ADA-based metric implementations.
- For best comparability, preprocess both sets identically (center-crop/resize policy is on you).
"""

import argparse
import os
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import scipy.linalg
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import functional as TF
from tqdm import tqdm

IMG_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}

INCEPTION_URL_DEFAULT = (
    "https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/"
    "inception-2015-12-05.pt"
)

# --------------------------
# Utils: list images
# --------------------------
def list_images(root: Path, recursive: bool) -> List[Path]:
    if recursive:
        files = [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMG_EXTS]
    else:
        files = [p for p in root.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS]
    files.sort()
    return files


# --------------------------
# Dataset: uint8 images
# --------------------------
class ImageFolderDataset(Dataset):
    def __init__(
        self,
        paths: List[Path],
        resize: Optional[int] = None,
        center_crop_long_edge: bool = False,
        resample: int = Image.Resampling.LANCZOS,
    ):
        self.paths = paths
        self.resize = resize
        self.center_crop_long_edge = center_crop_long_edge
        self.resample = resample

    def __len__(self) -> int:
        return len(self.paths)

    @staticmethod
    def _center_crop_long_edge(im: Image.Image) -> Image.Image:
        w, h = im.size
        s = min(w, h)
        left = (w - s) // 2
        top = (h - s) // 2
        return im.crop((left, top, left + s, top + s))

    def __getitem__(self, idx: int) -> torch.Tensor:
        p = self.paths[idx]
        with Image.open(p) as im:
            im = im.convert("RGB")

            if self.center_crop_long_edge:
                im = self._center_crop_long_edge(im)

            if self.resize is not None:
                im = im.resize((self.resize, self.resize), resample=self.resample)

            x = TF.to_tensor(im)  # float32 [0,1], CHW
            x = (x * 255.0).clamp(0, 255).to(torch.uint8)
            return x


# --------------------------
# Inception loading (TorchScript)
# --------------------------
_feature_detector_cache = {}

def load_stylegan_inception(detector_url: str, device: torch.device) -> torch.nn.Module:
    """
    Loads the StyleGAN2-ADA TorchScript Inception network.
    Requires internet for first download if URL is remote.
    """
    key = (detector_url, str(device))
    if key in _feature_detector_cache:
        return _feature_detector_cache[key]

    # NVLabs code loads via open_url; here we support both URL and local path.
    if detector_url.startswith("http://") or detector_url.startswith("https://"):
        # Download to a local cache file
        import urllib.request
        cache_dir = Path.home() / ".cache" / "fid_detectors"
        cache_dir.mkdir(parents=True, exist_ok=True)
        local_path = cache_dir / Path(detector_url).name
        if not local_path.exists():
            print(f"Downloading detector to {local_path} ...")
            urllib.request.urlretrieve(detector_url, local_path)
        detector_path = str(local_path)
    else:
        detector_path = detector_url

    detector = torch.jit.load(detector_path).eval().to(device)
    _feature_detector_cache[key] = detector
    return detector


# --------------------------
# Stats accumulator
# --------------------------
class FeatureStats:
    def __init__(self, max_items: Optional[int] = None):
        self.max_items = max_items
        self.num_items = 0
        self.num_features = None
        self.raw_mean = None
        self.raw_cov = None

    def _init(self, num_features: int):
        self.num_features = num_features
        self.raw_mean = np.zeros([num_features], dtype=np.float64)
        self.raw_cov = np.zeros([num_features, num_features], dtype=np.float64)

    def append(self, feats: np.ndarray):
        feats = np.asarray(feats, dtype=np.float32)
        assert feats.ndim == 2

        if self.max_items is not None and self.num_items >= self.max_items:
            return

        if self.max_items is not None and self.num_items + feats.shape[0] > self.max_items:
            feats = feats[: self.max_items - self.num_items]

        if self.num_features is None:
            self._init(feats.shape[1])
        else:
            assert feats.shape[1] == self.num_features

        self.num_items += feats.shape[0]
        f64 = feats.astype(np.float64)
        self.raw_mean += f64.sum(axis=0)
        self.raw_cov += f64.T @ f64

    def mean_cov(self) -> Tuple[np.ndarray, np.ndarray]:
        mean = self.raw_mean / self.num_items
        cov = self.raw_cov / self.num_items
        cov = cov - np.outer(mean, mean)
        return mean, cov


# --------------------------
# Compute features for a folder
# --------------------------
@torch.no_grad()
def compute_mu_sigma_for_folder(
    paths: List[Path],
    detector: torch.nn.Module,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    resize: Optional[int],
    center_crop_long_edge: bool,
    max_items: Optional[int],
) -> Tuple[np.ndarray, np.ndarray]:
    ds = ImageFolderDataset(paths, resize=resize, center_crop_long_edge=center_crop_long_edge)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    stats = FeatureStats(max_items=max_items)

    for batch in tqdm(dl, desc="Extract features", unit="batch"):
        # batch: uint8 CHW in [0,255]
        batch = batch.to(device, non_blocking=True)

        # StyleGAN Inception expects NCHW uint8 in [0,255]
        feats = detector(batch, return_features=True)  # (N, 2048) typically
        if isinstance(feats, (tuple, list)):
            feats = feats[0]
        feats_np = feats.detach().cpu().numpy()
        stats.append(feats_np)
        if stats.max_items is not None and stats.num_items >= stats.max_items:
            break

    return stats.mean_cov()


def fid_from_stats(mu1, sigma1, mu2, sigma2) -> float:
    m = np.square(mu1 - mu2).sum()
    s, _ = scipy.linalg.sqrtm(np.dot(sigma1, sigma2), disp=False)
    fid = np.real(m + np.trace(sigma1 + sigma2 - 2.0 * s))
    return float(fid)


def compute_fid_dir_to_dir(
    dir1: str,
    dir2: str,
    detector_url: str,
    device: str,
    batch_size: int,
    num_workers: int,
    resize: Optional[int],
    center_crop_long_edge: bool,
    recursive: bool,
    limit: Optional[int],
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

    n = min(len(paths1), len(paths2))
    if len(paths1) != len(paths2):
        print(f"Warning: unequal counts; truncating both to {n} images.")
        paths1 = paths1[:n]
        paths2 = paths2[:n]

    print(f"dir1: {len(paths1)} images | dir2: {len(paths2)} images")
    dev = torch.device(device)

    detector = load_stylegan_inception(detector_url, dev)

    print("Computing stats for dir1...")
    mu1, sigma1 = compute_mu_sigma_for_folder(
        paths1, detector, dev, batch_size, num_workers, resize, center_crop_long_edge, max_items=n
    )
    print("Computing stats for dir2...")
    mu2, sigma2 = compute_mu_sigma_for_folder(
        paths2, detector, dev, batch_size, num_workers, resize, center_crop_long_edge, max_items=n
    )

    return fid_from_stats(mu1, sigma1, mu2, sigma2)


def main():
    parser = argparse.ArgumentParser(description="Compute FID between two dirs using StyleGAN2-ADA Inception.")
    parser.add_argument("--dir1", type=str, required=True, help="Real images directory")
    parser.add_argument("--dir2", type=str, required=True, help="Fake images directory")
    parser.add_argument("--device", type=str, default="cuda", help="cuda, cuda:0, cpu")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--limit", type=int, default=None, help="Optional cap per dir (after sorting)")
    parser.add_argument("--resize", type=int, default=None, help="Optional resize NxN before features")
    parser.add_argument(
        "--center-crop-long-edge",
        action="store_true",
        help="Apply center-crop-to-square (min side) before resize. Matches common COCO prep.",
    )
    parser.add_argument("--detector", type=str, default=INCEPTION_URL_DEFAULT,
                        help="Path or URL to inception-2015-12-05.pt")
    args = parser.parse_args()

    fid = compute_fid_dir_to_dir(
        dir1=args.dir1,
        dir2=args.dir2,
        detector_url=args.detector,
        device=args.device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        resize=args.resize,
        center_crop_long_edge=args.center_crop_long_edge,
        recursive=args.recursive,
        limit=args.limit,
    )
    print(f"FID score: {fid}")


if __name__ == "__main__":
    main()