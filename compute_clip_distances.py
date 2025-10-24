#!/usr/bin/env python3
import argparse, json, os
from pathlib import Path

import torch
import torch.nn.functional as F
import pandas as pd
from transformers import CLIPModel, CLIPTokenizerFast

def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

@torch.no_grad()
def clip_text_features(model, tokenizer, texts, device, batch_size=64):
    """Return L2-normalized CLIP text features for a list of strings."""
    feats = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        enc = tokenizer(batch, padding=True, truncation=True, return_tensors="pt").to(device)
        f = model.get_text_features(**enc)               # already L2-normalized by HF
        f = F.normalize(f, dim=-1)                      # normalize again (safe)
        feats.append(f)
    return torch.cat(feats, dim=0)                      # [N, D]

def main():
    ap = argparse.ArgumentParser(description="Compute CLIP distances to target for synonyms and others per JSON file.")
    ap.add_argument("--input_dir", type=str, required=True, help="Folder with *.json files.")
    ap.add_argument("--output_csv", type=str, default="clip_prompt_distances.csv", help="Output CSV path.")
    ap.add_argument("--model_name", type=str, default="openai/clip-vit-large-patch14", help="Hugging Face CLIP checkpoint.")
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="cpu or cuda")
    ap.add_argument("--batch_size", type=int, default=64, help="Batch size for text encoding.")
    args = ap.parse_args()

    device = torch.device(args.device)
    tokenizer = CLIPTokenizerFast.from_pretrained(args.model_name)
    model = CLIPModel.from_pretrained(args.model_name).to(device).eval()

    rows = []
    input_dir = Path(args.input_dir)
    files = sorted(input_dir.glob("*.json"))
    if not files:
        raise SystemExit(f"No JSON files found in: {input_dir}")

    for fp in files:
        try:
            data = load_json(fp)
        except Exception as e:
            print(f"[WARN] Skipping {fp.name}: {e}")
            continue

        target = (data.get("target") or "").strip()
        if not target:
            print(f"[WARN] {fp.name}: empty target; skipping.")
            continue

        synonyms = [s.strip() for s in (data.get("synonyms") or []) if s and s.strip()]
        others   = [s.strip() for s in (data.get("other") or []) if s is not None and s.strip()]

        # de-duplicate while preserving order
        def uniq(seq):
            seen = set()
            out = []
            for x in seq:
                if x not in seen:
                    seen.add(x)
                    out.append(x)
            return out
        synonyms = uniq(synonyms)
        others   = uniq(others)

        # Build the batch: first is the target
        texts = [target] + synonyms + others
        feats = clip_text_features(model, tokenizer, texts, device, batch_size=args.batch_size)  # [1+S+O, D]
        tgt = feats[0:1]  # [1, D]
        rest = feats[1:]  # [S+O, D]

        # Cosine similarity with target (dot product; features are unit-norm)
        sims = (rest @ tgt.T).squeeze(1).cpu()  # [S+O]
        dists = (1.0 - sims)

        # Write rows for synonyms (first |synonyms| entries) then others
        idx = 0
        for s in synonyms:
            rows.append({
                "file": fp.name,
                "group": "synonym",
                "prompt": s,
                "cosine_similarity": float(sims[idx]),
                "cosine_distance": float(dists[idx]),
            })
            idx += 1
        for o in others:
            rows.append({
                "file": fp.name,
                "group": "other",
                "prompt": o,
                "cosine_similarity": float(sims[idx]),
                "cosine_distance": float(dists[idx]),
            })
            idx += 1

    if not rows:
        raise SystemExit("No rows produced—check your input files.")

    df = pd.DataFrame(rows).sort_values(["file", "group", "cosine_distance"])
    df.to_csv(args.output_csv, index=False)
    print(f"Wrote {len(df)} rows to {args.output_csv}")

    # Optional: print quick aggregates per file
    agg = df.groupby(["file", "group"])["cosine_distance"].agg(["mean", "std", "min", "max"]).reset_index()
    print(agg.to_string(index=False))

if __name__ == "__main__":
    main()
