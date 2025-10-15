#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, json, random, re
from pathlib import Path
from typing import List, Iterable

# --- CIFAR-100 fine labels ---
CIFAR100 = [
    "apple","aquarium_fish","baby","bear","beaver","bed","bee","beetle","bicycle",
    "bottle","bowl","boy","bridge","bus","butterfly","camel","can","castle","caterpillar",
    "cattle","chair","chimpanzee","clock","cloud","cockroach","couch","crab","crocodile",
    "cup","dinosaur","dolphin","elephant","flatfish","forest","fox","girl","hamster",
    "house","kangaroo","computer_keyboard","lamp","lawn_mower","leopard","lion","lizard",
    "lobster","man","maple_tree","motorcycle","mountain","mouse","mushroom","oak_tree",
    "orange","orchid","otter","palm_tree","pear","pickup_truck","pine_tree","plain",
    "plate","poppy","porcupine","possum","rabbit","raccoon","ray","road","rocket","rose",
    "sea","seal","shark","shrew","skunk","skyscraper","snail","snake","spider","squirrel",
    "streetcar","sunflower","sweet_pepper","table","tank","telephone","television","tiger",
    "tractor","train","trout","tulip","turtle","wardrobe","whale","willow_tree","wolf",
    "woman","worm"
]

# Per your spec (note: "sea" appears twice because your list did)
REFERENCE_CHOICES = ["forest", "sea", "street", "ground", "sea", "sky"]

# Curated alternates ONLY for labels present in CIFAR-100.
CURATED = {
    "dog": ["canine", "hound", "pooch"],
    "cat": ["feline", "tomcat", "kitty"],
    "truck": ["lorry", "rig", "hauler"],
    "pickup_truck": ["pickup", "ute", "light truck"],
    "automobile": ["car", "motorcar", "auto"],
    "airplane": ["aircraft", "plane", "aeroplane"],
    "ship": ["vessel", "boat", "seacraft"],
    "train": ["railway train", "locomotive", "railcar"],
    "television": ["tv", "television set", "receiver"],
    "telephone": ["phone", "handset", "telephone set"],
    "chair": ["seat", "armchair", "stool"],
    "table": ["desk", "worktable", "dining table"],
    "couch": ["sofa", "settee", "lounge"],
    "tiger": ["big cat", "striped feline", "panthera tigris"],
    "lion": ["big cat", "panthera leo", "male lion"],
    "leopard": ["panthera pardus", "spotted feline", "big cat"],
    "wolf": ["canis lupus", "wild dog", "timber wolf"],
    "whale": ["cetacean", "sea mammal", "baleen whale"],
    "dolphin": ["cetacean", "marine mammal", "porpoise"],
    "elephant": ["pachyderm", "asian elephant", "african elephant"],
    "rabbit": ["bunny", "hare", "lagomorph"],
    "mouse": ["rodent", "house mouse", "field mouse"],
    "bee": ["honeybee", "bumblebee", "apis"],
    "butterfly": ["lepidopteran", "nymphalid", "swallowtail"],
    "snake": ["serpent", "colubrid", "viper"],
    "spider": ["arachnid", "tarantula", "orb weaver"],
    "snail": ["gastropod", "land snail", "garden snail"],
    "shark": ["selachian", "great white", "hammerhead"],
    "ray": ["batoid", "manta ray", "stingray"],
    "trout": ["salmonid", "brown trout", "rainbow trout"],
    "turtle": ["cheloniid", "sea turtle", "freshwater turtle"],
    "flatfish": ["flounder", "halibut", "sole"],
    "camel": ["dromedary", "bactrian camel", "camelid"],
    "cattle": ["bovine", "cow", "bull"],
    "chimpanzee": ["pan troglodytes", "chimp", "great ape"],
    "kangaroo": ["macropod", "roo", "red kangaroo"],
    "hamster": ["rodent", "golden hamster", "syrian hamster"],
    "lobster": ["decapod", "clawed lobster", "homarus"],
    "crab": ["decapod", "shore crab", "blue crab"],
    "crocodile": ["crocodylid", "croc", "niloticus"],
    "lizard": ["reptile", "gecko", "iguana"],
    "fox": ["vulpine", "red fox", "vulpes"],
    "bear": ["ursid", "brown bear", "polar bear"],
    "beaver": ["rodent", "castor", "north american beaver"],
    "skunk": ["mephitid", "striped skunk", "mephitis"],
    "porcupine": ["erethizon", "rodent", "hystricid"],
    "possum": ["opossum", "didelphid", "virginia opossum"],
    "raccoon": ["procyonid", "procyon lotor", "trash panda"],
    "otter": ["mustelid", "sea otter", "river otter"],
    "seal": ["pinniped", "earless seal", "harbor seal"],
    "orchid": ["orchidaceae", "phalaenopsis", "cattleya"],
    "tulip": ["liliaceous", "tulipa", "flowering tulip"],
    "sunflower": ["helianthus", "flower head", "sunflower bloom"],
    "maple_tree": ["acer", "maple", "maple sapling"],
    "oak_tree": ["quercus", "oak", "oak sapling"],
    "willow_tree": ["salix", "willow", "weeping willow"],
    "pine_tree": ["pinus", "pine", "conifer"],
    "palm_tree": ["arecaceae", "palm", "date palm"],
    "mushroom": ["agaric", "toadstool", "basidiomycete"],
    "sweet_pepper": ["bell pepper", "capsicum", "paprika"],
    "orange": ["citrus", "navel orange", "blood orange"],
    "pear": ["pyrus", "bartlett pear", "anjou pear"],
    "apple": ["malus", "pippin", "granny smith"],
    "rose": ["rosa", "rose bloom", "garden rose"],
    "poppy": ["papaver", "opium poppy", "corn poppy"],
    "computer_keyboard": ["pc keyboard", "qwerty keyboard", "mechanical keyboard"],
    "lawn_mower": ["mower", "push mower", "ride-on mower"],
    "motorcycle": ["motorbike", "bike", "two-wheeler"],
    "streetcar": ["tram", "trolley", "light rail"],
    "skyscraper": ["high-rise", "tower block", "office tower"],
    "wardrobe": ["closet", "armoire", "clothes press"],
    "tank": ["armored vehicle", "battle tank", "mbt"],
    "rocket": ["space rocket", "launch vehicle", "booster"],
    "road": ["highway", "street", "paved road"],
    "bridge": ["span", "overpass", "arch bridge"],
    "castle": ["fortress", "keep", "stronghold"],
    "house": ["home", "residence", "dwelling"],
    "bed": ["single bed", "double bed", "bunk bed"],
    "bottle": ["glass bottle", "plastic bottle", "flask"],
    "bowl": ["dish", "serving bowl", "soup bowl"],
    "plate": ["dinner plate", "dish", "charger plate"],
    "cup": ["teacup", "mug", "coffee cup"],
    "lamp": ["desk lamp", "table lamp", "light"],
    "clock": ["timepiece", "wall clock", "alarm clock"],
    "television": ["tv", "television set", "receiver"],
    "telephone": ["phone", "handset", "telephone set"],
    "computer_keyboard": ["pc keyboard", "qwerty keyboard", "mechanical keyboard"],
}

def a_or_an(text: str) -> str:
    w = text.strip().lower()
    return "an" if re.match(r"^(?:[aeiou]|hour|honest|heir)", w) else "a"

def phrase(x: str) -> str:
    x = x.replace("_", " ")
    return f"{a_or_an(x)} photo of the {x}"

def normalize_candidates(words: Iterable[str], base: str) -> List[str]:
    """Cleanup WordNet/curated outputs: lowercase, de-underscore, remove dup/identical, short & nouny."""
    seen, out = set(), []
    base_l = base.lower().replace("_", " ")
    for w in words:
        w2 = str(w).lower().replace("_", " ").strip()
        if not w2 or w2 == base_l:
            continue
        # keep 1–3 words, alphabetic-ish, not too long to stay prompty
        if len(w2.split()) > 3 or len(w2) > 24:
            continue
        if not re.match(r"^[a-z][a-z\s\-]+$", w2):
            continue
        if w2 in seen:
            continue
        seen.add(w2)
        out.append(w2)
    return out

def propose_synonyms(label: str, k: int = 3) -> List[str]:
    # 1) curated if present
    if label in CURATED:
        cands = normalize_candidates(CURATED[label], label)
        return [phrase(w) for w in cands[:k]]

    # 2) WordNet fallback (optional)
    try:
        from nltk.corpus import wordnet as wn  # type: ignore
        try:
            wn.ensure_loaded()
        except Exception:
            import nltk  # type: ignore
            nltk.download("wordnet", quiet=True)
            wn.ensure_loaded()

        base = label.replace("_", " ")
        raw = set()

        for ss in wn.synsets(base, pos="n"):
            for lem in ss.lemmas():
                raw.add(lem.name())
            # add hypernyms and hyponyms to get category-fitting words
            for rel in (ss.hypernyms() + ss.hyponyms()):
                for lem in rel.lemmas():
                    raw.add(lem.name())

        cands = normalize_candidates(raw, label)
        if cands:
            return [phrase(w) for w in cands[:k]]
    except Exception:
        pass

    # 3) If nothing found, just reuse the label itself with mild variants (but still exact format)
    base = label.replace("_", " ")
    fallbacks = [base, base, base]  # keep it strict and simple
    return [phrase(w) for w in fallbacks[:k]]

def make_entry(target: str, labels: List[str], rng: random.Random, others_count: int):
    reference = rng.choice(REFERENCE_CHOICES)
    pool = [c for c in labels if c != target]
    rng.shuffle(pool)
    others = pool[:others_count]
    return {
        "target": phrase(target),
        "reference": phrase(reference),
        "synonyms": propose_synonyms(target, 3),
        "other": [phrase(x) for x in others],
    }

def main():
    ap = argparse.ArgumentParser(description="Generate CIFAR-100 prompt JSONs.")
    ap.add_argument("--out_dir", type=Path, default=Path("cifar100_prompts"))
    ap.add_argument("--others", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    labels = CIFAR100[:]  # copy
    rng = random.Random(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # per-class files
    for lbl in labels:
        data = make_entry(lbl, labels, rng, args.others)
        with open(args.out_dir / f"{lbl}.json", "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    # combined convenience file
    combined = {lbl: make_entry(lbl, labels, rng, args.others) for lbl in labels}
    with open(args.out_dir / "all_prompts.json", "w", encoding="utf-8") as f:
        json.dump(combined, f, ensure_ascii=False, indent=2)

    print(f"Wrote {len(labels)} JSONs + all_prompts.json to {args.out_dir}")

if __name__ == "__main__":
    main()
