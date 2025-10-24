#!/usr/bin/env python3
import argparse, json, re
from pathlib import Path

import torch
import torch.nn.functional as F
import pandas as pd
from transformers import CLIPModel, CLIPTokenizerFast

AUG_TEMPLATES = [
    "{} in a photo",
    "{} in a snapshot",
    "A snapshot of {}",
    "A photograph showcasing {}",
    "An illustration of {}",
    "A digital rendering of {}",
    "A visual representation of {}",
    "A graphic of {}",
    "A shot of {}",
    "A photo of {}",
    "A black and white image of {}",
    "A depiction in portrait form of {}",
    "A scene depicting {} during a public gathering",
    "{} captured in an image",
    "A depiction created with oil paints capturing {}",
    "An image of {}",
    "A drawing capturing the essence of {}",
    "An official photograph featuring {}",
    "A detailed sketch of {}",
    "{} during sunset/sunrise",
    "{} in a detailed portrait",
    "An official photo of {}",
    "Historic photo of {}",
    "Detailed portrait of {}",
    "A painting of {}",
    "HD picture of {}",
    "Magazine cover capturing {}",
    "Painting-like image of {}",
    "Hand-drawn art of {}",
    "An oil portrait of {}",
    "{} in a sketch painting",
]

def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

def last_word(text: str) -> str:
    # take the last alphanumeric-ish token
    toks = re.findall(r"[A-Za-z0-9\-]+", text.strip().lower())
    return toks[-1] if toks else ""

@torch.no_grad()
def clip_text_features(model, tokenizer, texts, device, batch_size=64):
    """Return L2-normalized CLIP text features for a list of strings."""
    feats = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        enc = tokenizer(batch, padding=True, truncation=True, return_tensors="pt").to(device)
        f = model.get_text_features(**enc)   # HF returns L2-normalized
        f = F.normalize(f, dim=-1)
        feats.append(f)
    return torch.cat(feats, dim=0)  # [N, D]

@torch.no_grad()
def augmented_embedding(model, tokenizer, content: str, device, batch_size=64):
    """
    Build prompts by filling templates with `content`, encode, mean-pool in feature space,
    then L2-normalize. Returns [D] (float).
    """
    if not content:
        return None
    prompts = [tpl.format(content) for tpl in AUG_TEMPLATES]
    feats = clip_text_features(model, tokenizer, prompts, device, batch_size=batch_size)  # [T,D]
    emb = feats.mean(dim=0, keepdim=True)                      # [1,D]
    emb = F.normalize(emb, dim=-1).squeeze(0)                  # [D]
    return emb

def main():
    ap = argparse.ArgumentParser(description="Compute CLIP distances to target using augmentation-based embeddings per JSON file.")
    ap.add_argument("--input_dir", type=str, required=True, help="Folder with *.json files.")
    ap.add_argument("--output_csv", type=str, default="clip_prompt_distances_aug.csv", help="Per-prompt output CSV.")
    ap.add_argument("--summary_csv", type=str, default="clip_prompt_means_aug.csv", help="Per-file means CSV.")
    ap.add_argument("--target_aug_csv", type=str, default="clip_target_aug_distances.csv", help="Per-augmentation distances of target vs its own augmentations.")
    ap.add_argument("--model_name", type=str, default="openai/clip-vit-large-patch14", help="Hugging Face CLIP checkpoint.")
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="cpu or cuda")
    ap.add_argument("--batch_size", type=int, default=64, help="Batch size for text encoding.")
    args = ap.parse_args()

    device = torch.device(args.device)
    tokenizer = CLIPTokenizerFast.from_pretrained(args.model_name)
    model = CLIPModel.from_pretrained(args.model_name).to(device).eval()

    rows = []          # synonyms/others vs target (as before)
    target_aug_rows = []  # NEW: per-target augmentation distances to the original target prompt

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

        # Build augmented embedding for TARGET (using its last word) - for comparisons with synonyms/others
        tgt_content = last_word(target)
        tgt_emb = augmented_embedding(model, tokenizer, tgt_content, device, args.batch_size)
        if tgt_emb is None:
            print(f"[WARN] {fp.name}: target produced empty content; skipping.")
            continue

        # --- NEW: distances from the original target prompt to each of its own augmentations ---
        # Original target prompt embedding (canonical CLIP text feature)
        tgt_direct_feat = clip_text_features(model, tokenizer, [target], device, batch_size=args.batch_size)[0]  # [D]
        # All augmented prompts for the target's last word
        tgt_aug_prompts = [tpl.format(tgt_content) for tpl in AUG_TEMPLATES]
        tgt_aug_feats = clip_text_features(model, tokenizer, tgt_aug_prompts, device, batch_size=args.batch_size)  # [T,D]
        # Cosine similarity/distances to the original target prompt
        sims_aug = (tgt_aug_feats @ tgt_direct_feat)  # [T], features are L2-normalized
        dists_aug = (1.0 - sims_aug).tolist()
        for aug_text, sim_val, dist_val in zip(tgt_aug_prompts, sims_aug.tolist(), dists_aug):
            target_aug_rows.append({
                "file": fp.name,
                "group": "target_aug",
                "target": target,
                "content_last_word": tgt_content,
                "aug_prompt": aug_text,
                "cosine_similarity": float(sim_val),
                "cosine_distance": float(dist_val),
            })
        # ---------------------------------------------------------------

        # Per-prompt distances (synonyms, others) using their OWN last word content
        for group_name, prompt_list in (("synonym", synonyms), ("other", others)):
            for p in prompt_list:
                content = last_word(p)
                emb = augmented_embedding(model, tokenizer, content, device, args.batch_size)
                if emb is None:
                    continue
                sim = float(torch.dot(tgt_emb, emb))   # both L2-normalized → cosine similarity
                dist = 1.0 - sim
                rows.append({
                    "file": fp.name,
                    "group": group_name,
                    "prompt": p,
                    "content_last_word": content,
                    "cosine_similarity": sim,
                    "cosine_distance": dist,
                })

    if not rows and not target_aug_rows:
        raise SystemExit("No rows produced—check your input files.")

    # --- Write per-prompt distances for synonyms/others (unchanged) ---
    if rows:
        df = pd.DataFrame(rows)
        df = df.sort_values(["file", "group", "cosine_distance"])
        df.to_csv(args.output_csv, index=False)
        print(f"Wrote per-prompt distances to {args.output_csv}")
    else:
        df = pd.DataFrame(columns=["file","group","prompt","content_last_word","cosine_similarity","cosine_distance"])

    # --- Write per-augmentation distances for target vs its own augmentations (NEW) ---
    if target_aug_rows:
        df_taug = pd.DataFrame(target_aug_rows).sort_values(["file","cosine_distance"])
        df_taug.to_csv(args.target_aug_csv, index=False)
        print(f"Wrote target augmentation distances to {args.target_aug_csv}")
    else:
        df_taug = pd.DataFrame(columns=["file","group","target","content_last_word","aug_prompt","cosine_similarity","cosine_distance"])

    # --- Compute per-file means for synonyms/others + target_aug (NEW in summary) ---
    df_all = pd.concat([
        df[["file","group","cosine_distance"]],
        df_taug[["file","group","cosine_distance"]],
    ], ignore_index=True)

    if not df_all.empty:
        means = (
            df_all.groupby(["file", "group"])["cosine_distance"]
                  .agg(mean="mean", std="std", min="min", max="max")
                  .reset_index()
                  .pivot(index="file", columns="group")
        )
        # flatten columns
        means.columns = [f"{g}_{stat}" for g, stat in means.columns]
        means = means.reset_index()
        means.to_csv(args.summary_csv, index=False)
        print(f"Wrote per-file means to {args.summary_csv}")
        print(means.to_string(index=False))
    else:
        print("No summary generated (no rows).")

if __name__ == "__main__":
    main()
