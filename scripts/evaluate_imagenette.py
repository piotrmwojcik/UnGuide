#!/usr/bin/env python3
"""
Evaluate Imagenette ablation study results.

Metrics:
  - UA  (Unlearning Accuracy): fraction of target-concept images NOT classified
         as the target class by CLIP zero-shot  (higher = better unlearning)
  - RA  (Retain Accuracy):     CLIP zero-shot accuracy on retain/general prompts
         (higher = better knowledge preservation)
  - CS  (CLIP Score):          mean cosine similarity between generated image and
         its prompt  (overall generation quality)

Outputs a JSON + prints a LaTeX-ready table.
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from PIL import Image
from tqdm import tqdm

# Imagenette classes
TARGET_CONCEPTS = [
    "tench", "English springer", "cassette player", "chain saw", "church",
    "French horn", "garbage truck", "gas pump", "golf ball", "parachute",
]

RETAIN_CONCEPTS = [
    "cat", "car", "tree", "house", "bicycle",
    "flower", "mountain", "river", "airplane", "boat",
]

ALL_CANDIDATE_LABELS = TARGET_CONCEPTS + RETAIN_CONCEPTS

EXPERIMENT_DIRS = {
    "ht1":           "imagenette_sd_alpha1_strong100_ht1",
    "ht1_neutral":   "imagenette_sd_alpha1_strong100_ht1_neutral",
    "ht100":         "imagenette_sd_alpha1_strong100_ht100",
    "ht100_neutral": "imagenette_sd_alpha1_strong100_ht100_neutral",
    "ht300":         "imagenette_sd_alpha1_strong100_ht300",
    "ht300_neutral": "imagenette_sd_alpha1_strong100_ht300_neutral",
    "ht500":         "imagenette_sd_alpha1_strong100_ht500",
    "ht500_neutral": "imagenette_sd_alpha1_strong100_ht500_neutral",
}


def load_clip(device):
    """Load CLIP model for evaluation."""
    try:
        import open_clip
        model, _, preprocess = open_clip.create_model_and_transforms(
            "ViT-L-14", pretrained="openai", device=device
        )
        tokenizer = open_clip.get_tokenizer("ViT-L-14")
        return model, preprocess, tokenizer, "open_clip"
    except ImportError:
        pass

    import transformers
    from transformers import CLIPModel, CLIPProcessor
    clip = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device).eval()
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
    return clip, processor, None, "hf_clip"


@torch.no_grad()
def clip_classify(images, labels, model, preprocess, tokenizer, backend, device):
    """Zero-shot CLIP classification. Returns predicted label index per image."""
    if backend == "open_clip":
        import open_clip
        text_tokens = tokenizer(["a photo of a " + l for l in labels]).to(device)
        text_feats = model.encode_text(text_tokens)
        text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)

        preds = []
        for img in images:
            img_t = preprocess(img).unsqueeze(0).to(device)
            img_feat = model.encode_image(img_t)
            img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
            sim = (img_feat @ text_feats.T).squeeze(0)
            preds.append(sim.argmax().item())
        return preds
    else:
        # HuggingFace CLIP
        processor = preprocess
        preds = []
        text_inputs = processor(
            text=["a photo of a " + l for l in labels],
            return_tensors="pt", padding=True
        ).to(device)
        text_feats = model.get_text_features(**text_inputs)
        text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)

        for img in images:
            img_inputs = processor(images=img, return_tensors="pt").to(device)
            img_feat = model.get_image_features(**img_inputs)
            img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
            sim = (img_feat @ text_feats.T).squeeze(0)
            preds.append(sim.argmax().item())
        return preds


@torch.no_grad()
def clip_score(images, prompts, model, preprocess, tokenizer, backend, device):
    """Mean CLIP cosine similarity between images and their prompts."""
    scores = []
    if backend == "open_clip":
        for img, prompt in zip(images, prompts):
            img_t = preprocess(img).unsqueeze(0).to(device)
            text_t = tokenizer([prompt]).to(device)
            img_feat = model.encode_image(img_t)
            text_feat = model.encode_text(text_t)
            img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
            text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)
            scores.append((img_feat @ text_feat.T).item())
    else:
        processor = preprocess
        for img, prompt in zip(images, prompts):
            inputs = processor(text=[prompt], images=img, return_tensors="pt", padding=True).to(device)
            outputs = model(**inputs)
            scores.append(outputs.logits_per_image.item() / 100.0)
    return scores


def collect_images(gen_dir):
    """Walk generated_images/ directory and return {class_name: [PIL.Image, ...]}."""
    result = {}
    if not os.path.isdir(gen_dir):
        return result
    for subdir in os.listdir(gen_dir):
        subdir_path = os.path.join(gen_dir, subdir)
        if not os.path.isdir(subdir_path):
            continue
        imgs = []
        for fname in sorted(os.listdir(subdir_path)):
            if fname.lower().endswith((".png", ".jpg", ".jpeg")):
                try:
                    imgs.append(Image.open(os.path.join(subdir_path, fname)).convert("RGB"))
                except Exception:
                    continue
        if imgs:
            result[subdir] = imgs
    return result


def evaluate_experiment(gen_dir, model, preprocess, tokenizer, backend, device):
    """Evaluate a single experiment directory."""
    images_by_class = collect_images(gen_dir)
    if not images_by_class:
        return None

    n_target = len(TARGET_CONCEPTS)

    # UA: for target-concept images, check if CLIP classifies them as NOT the target
    ua_correct = 0
    ua_total = 0
    target_clip_scores = []

    for concept in TARGET_CONCEPTS:
        # Match class subdirectory (last word of concept)
        class_key = concept.split()[-1].lower()
        matched_imgs = None
        for k, v in images_by_class.items():
            if k.lower() == class_key:
                matched_imgs = v
                break
        if matched_imgs is None:
            continue

        preds = clip_classify(matched_imgs, ALL_CANDIDATE_LABELS, model, preprocess, tokenizer, backend, device)
        concept_idx = TARGET_CONCEPTS.index(concept)
        for pred in preds:
            ua_total += 1
            if pred != concept_idx:  # NOT classified as the target = unlearned
                ua_correct += 1

        # CLIP score for target concept images
        prompts = [f"a photo of a {concept}"] * len(matched_imgs)
        scores = clip_score(matched_imgs, prompts, model, preprocess, tokenizer, backend, device)
        target_clip_scores.extend(scores)

    ua = ua_correct / max(1, ua_total)

    # RA: for retain-concept images, check if CLIP still classifies them correctly
    ra_correct = 0
    ra_total = 0
    retain_clip_scores = []

    for concept in RETAIN_CONCEPTS:
        class_key = concept.split()[-1].lower()
        matched_imgs = None
        for k, v in images_by_class.items():
            if k.lower() == class_key:
                matched_imgs = v
                break
        if matched_imgs is None:
            continue

        concept_idx = ALL_CANDIDATE_LABELS.index(concept)
        preds = clip_classify(matched_imgs, ALL_CANDIDATE_LABELS, model, preprocess, tokenizer, backend, device)
        for pred in preds:
            ra_total += 1
            if pred == concept_idx:
                ra_correct += 1

        prompts = [f"a photo of a {concept}"] * len(matched_imgs)
        scores = clip_score(matched_imgs, prompts, model, preprocess, tokenizer, backend, device)
        retain_clip_scores.extend(scores)

    ra = ra_correct / max(1, ra_total)

    # Overall CLIP score
    all_scores = target_clip_scores + retain_clip_scores

    return {
        "UA": round(ua, 4),
        "UA_n": ua_total,
        "RA": round(ra, 4),
        "RA_n": ra_total,
        "CS_target": round(np.mean(target_clip_scores), 4) if target_clip_scores else None,
        "CS_retain": round(np.mean(retain_clip_scores), 4) if retain_clip_scores else None,
        "CS_all": round(np.mean(all_scores), 4) if all_scores else None,
    }


def print_table(results):
    """Print a markdown + LaTeX table of results."""
    print("\n## Ablation Results: Imagenette SD Alpha1 Strong100\n")

    # Markdown table
    print("| HT Steps | Mapping  | UA     | RA     | CS(target) | CS(retain) |")
    print("|----------|----------|--------|--------|------------|------------|")

    for tag in ["ht1", "ht100", "ht300", "ht500"]:
        for suffix, mapping_label in [("", "Original"), ("_neutral", "Neutral")]:
            key = f"{tag}{suffix}"
            r = results.get(key)
            if r is None:
                print(f"| {tag.replace('ht',''): <8} | {mapping_label: <8} | -      | -      | -          | -          |")
                continue
            ht_val = tag.replace("ht", "")
            print(f"| {ht_val: <8} | {mapping_label: <8} | {r['UA']:.4f} | {r['RA']:.4f} | {r.get('CS_target', '-'): <10} | {r.get('CS_retain', '-'): <10} |")

    # LaTeX table
    print("\n% LaTeX table")
    print(r"\begin{tabular}{cc|cccc}")
    print(r"\toprule")
    print(r"HT Steps & Mapping & UA $\uparrow$ & RA $\uparrow$ & CS(target) $\downarrow$ & CS(retain) $\uparrow$ \\")
    print(r"\midrule")

    for tag in ["ht1", "ht100", "ht300", "ht500"]:
        for suffix, mapping_label in [("", "Original"), ("_neutral", "Neutral")]:
            key = f"{tag}{suffix}"
            r = results.get(key)
            if r is None:
                print(f"{tag.replace('ht','')} & {mapping_label} & - & - & - & - \\\\")
                continue
            ht_val = tag.replace("ht", "")
            cs_t = f"{r['CS_target']:.4f}" if r.get('CS_target') is not None else "-"
            cs_r = f"{r['CS_retain']:.4f}" if r.get('CS_retain') is not None else "-"
            print(f"{ht_val} & {mapping_label} & {r['UA']:.4f} & {r['RA']:.4f} & {cs_t} & {cs_r} \\\\")

    print(r"\bottomrule")
    print(r"\end{tabular}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate Imagenette ablation study")
    parser.add_argument("--output-base", type=str, default="output",
                        help="Base output directory containing experiment subdirs")
    parser.add_argument("--results-file", type=str, default="output/ablation_imagenette_results.json",
                        help="Where to save JSON results")
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("Loading CLIP model for evaluation...")
    model, preprocess, tokenizer, backend = load_clip(device)
    print(f"CLIP backend: {backend}")

    results = {}
    for tag, dirname in EXPERIMENT_DIRS.items():
        gen_dir = os.path.join(args.output_base, dirname, "generated_images")
        print(f"\n--- Evaluating: {tag} ({gen_dir}) ---")

        if not os.path.isdir(gen_dir):
            print(f"  Skipping (directory not found)")
            continue

        metrics = evaluate_experiment(gen_dir, model, preprocess, tokenizer, backend, device)
        if metrics is None:
            print(f"  Skipping (no images found)")
            continue

        results[tag] = metrics
        print(f"  UA={metrics['UA']:.4f} (n={metrics['UA_n']}), "
              f"RA={metrics['RA']:.4f} (n={metrics['RA_n']}), "
              f"CS_target={metrics.get('CS_target', '-')}, "
              f"CS_retain={metrics.get('CS_retain', '-')}")

    # Save results
    os.makedirs(os.path.dirname(args.results_file), exist_ok=True)
    with open(args.results_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {args.results_file}")

    print_table(results)


if __name__ == "__main__":
    main()
