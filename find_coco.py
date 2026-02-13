#!/usr/bin/env python3
import os
import csv
import shutil
from argparse import ArgumentParser

def coco_filename(coco_id: int) -> str:
    # COCO 2014 files are 12-digit zero-padded .jpg (e.g., COCO_val2014_000000244215.jpg)
    return f"COCO_val2014_{coco_id:012d}.jpg"

def main(csv_path: str, val_dir: str, out_dir: str, limit: int = 10000):
    os.makedirs(out_dir, exist_ok=True)

    copied = 0
    missing = 0

    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if copied >= limit:
                break

            coco_id_str = (row.get("coco_id") or "").strip()
            if not coco_id_str.isdigit():
                continue

            coco_id = int(coco_id_str)
            fname = coco_filename(coco_id)
            src = os.path.join(val_dir, fname)
            dst = os.path.join(out_dir, fname)

            if not os.path.exists(src):
                print('!!! missing ', coco_id_str)
                missing += 1
                continue

            shutil.copy2(src, dst)
            copied += 1

    print(f"Copied:  {copied}")
    print(f"Missing: {missing}")
    print(f"Output:  {out_dir}")

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--csv_path", required=True, help="CSV with columns including coco_id")
    parser.add_argument("--val_dir", default="/home/pwojcik/UnGuide/coco30_bck/val2014",
                        help="Path to COCO val2014 images directory")
    parser.add_argument("--out_dir", required=True, help="Where to copy selected images")
    parser.add_argument("--limit", type=int, default=10000, help="How many images to copy (default 10000)")
    args = parser.parse_args()

    main(args.csv_path, args.val_dir, args.out_dir, args.limit)
