#!/usr/bin/env python3
import os
import csv
import re
import shutil
from argparse import ArgumentParser

INT_RE = re.compile(r"\d+")

def coco_filename(coco_id: int) -> str:
    # COCO 2014 files are 12-digit zero-padded .jpg (e.g., COCO_val2014_000000244215.jpg)
    return f"COCO_val2014_{coco_id:012d}.jpg"

def parse_int(value) -> int | None:
    """
    Extract the first integer found in the value.
    Returns None if no digits are found.
    Handles values like '42^M', '  244215\\r', 'coco_id=244215', etc.
    """
    if value is None:
        return None
    s = str(value).strip()
    m = INT_RE.search(s)
    if not m:
        return None
    return int(m.group(0))

def main(csv_path: str, val_dir: str, out_dir: str, limit: int = 10000, id_col: str = "coco_id"):
    os.makedirs(out_dir, exist_ok=True)

    copied = 0
    missing = 0
    skipped = 0

    with open(csv_path, newline="", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)

        # Help if user passed wrong column name
        if reader.fieldnames and id_col not in reader.fieldnames:
            print(f"Warning: '{id_col}' not found in CSV header. Available columns: {reader.fieldnames}")

        for row in reader:
            if copied >= limit:
                break

            coco_id = parse_int(row.get(id_col))
            if coco_id is None:
                skipped += 1
                continue

            fname = coco_filename(coco_id)
            src = os.path.join(val_dir, fname)
            dst = os.path.join(out_dir, fname)

            if not os.path.exists(src):
                print(f"!!! missing {coco_id} -> {src}")
                missing += 1
                continue

            shutil.copy2(src, dst)
            copied += 1

    print(f"Copied:  {copied}")
    print(f"Missing: {missing}")
    print(f"Skipped (no id): {skipped}")
    print(f"Output:  {out_dir}")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--csv_path", required=True, help="CSV with a COCO id column")
    parser.add_argument("--val_dir", default="/home/pwojcik/UnGuide/coco30_bck/val2014",
                        help="Path to COCO val2014 images directory")
    parser.add_argument("--out_dir", required=True, help="Where to copy selected images")
    parser.add_argument("--limit", type=int, default=10000, help="How many images to copy")
    parser.add_argument("--id_col", default="coco_id", help="Column name that contains the COCO image id")
    args = parser.parse_args()

    main(args.csv_path, args.val_dir, args.out_dir, args.limit, args.id_col)
