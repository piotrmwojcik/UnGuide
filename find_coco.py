#!/usr/bin/env python3
import os
import csv
import re
import shutil
from argparse import ArgumentParser

INT_RE = re.compile(r"\d+")

def coco_filename(coco_id: int) -> str:
    return f"COCO_val2014_{coco_id:012d}.jpg"

def parse_int(value):
    if value is None:
        return None
    s = str(value).strip()
    m = INT_RE.search(s)
    if not m:
        return None
    return int(m.group(0))

def main(csv_path: str, val_dir: str, out_dir: str, limit: int = 10000,
         id_col: str = "coco_id", id_col_idx: int | None = None):
    os.makedirs(out_dir, exist_ok=True)

    copied = 0
    missing = 0
    skipped = 0

    with open(csv_path, newline="", encoding="utf-8", errors="replace") as f:
        # Peek first row
        reader0 = csv.reader(f)
        first = next(reader0, None)
        if first is None:
            print("Empty CSV.")
            return

        # Decide header vs no-header
        has_header = any(cell.strip() == id_col for cell in first)

        # Rewind
        f.seek(0)

        if has_header:
            reader = csv.DictReader(f)
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
        else:
            # Headerless: use index (default last column)
            reader = csv.reader(f)
            for row in reader:
                if copied >= limit:
                    break
                if not row:
                    skipped += 1
                    continue

                idx = id_col_idx if id_col_idx is not None else (len(row) - 1)
                if idx < 0 or idx >= len(row):
                    skipped += 1
                    continue

                coco_id = parse_int(row[idx])
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
    parser.add_argument("--csv_path", required=True)
    parser.add_argument("--val_dir", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--limit", type=int, default=10000)
    parser.add_argument("--id_col", default="coco_id", help="Column name if CSV has header")
    parser.add_argument("--id_col_idx", type=int, default=None,
                        help="Column index for headerless CSV (0-based). Default: last column.")
    args = parser.parse_args()

    main(args.csv_path, args.val_dir, args.out_dir, args.limit, args.id_col, args.id_col_idx)
