import random

import nltk
from nltk.corpus import wordnet as wn


def setup_wordnet():
    try:
        wn.synsets("test")
    except LookupError:
        nltk.download("wordnet", quiet=True)
        nltk.download("omw-1.4", quiet=True)


def get_visual_synonyms_for_concept(concept_name, max_synonyms=7):
    synonyms = set()
    clean_concept = concept_name.replace("_", " ").lower()
    synsets = wn.synsets(clean_concept.replace(" ", "_"), pos="n")

    if not synsets:
        for word in clean_concept.split():
            synsets.extend(wn.synsets(word, pos="n"))

    if not synsets:
        return []

    primary_synset = synsets[0]

    for lemma in primary_synset.lemmas():
        synonym = lemma.name().replace("_", " ")
        if synonym.lower() != clean_concept.lower():
            synonyms.add(synonym)

    for hyponym in primary_synset.hyponyms()[:10]:
        for lemma in hyponym.lemmas()[:1]:
            synonym = lemma.name().replace("_", " ")
            if len(synonym.split()) <= 2 and synonym.lower() != clean_concept.lower():
                synonyms.add(synonym)

    for hypernym in primary_synset.hypernyms()[:1]:
        for sister in hypernym.hyponyms()[:8]:
            if sister != primary_synset:
                for lemma in sister.lemmas()[:1]:
                    synonym = lemma.name().replace("_", " ")
                    if (
                        len(synonym.split()) <= 2
                        and synonym.lower() != clean_concept.lower()
                    ):
                        synonyms.add(synonym)

    synonym_list = list(synonyms)
    random.shuffle(synonym_list)
    return synonym_list[:max_synonyms]


def generate_synonyms_for_list(
    concept_list, synonyms_per_concept=7, relevance_threshold=3
):
    setup_wordnet()
    results = {}

    for concept in concept_list:
        synonyms = get_visual_synonyms_for_concept(
            concept, synonyms_per_concept, relevance_threshold
        )
        results[concept] = synonyms
        print(f"'{concept}': {synonyms}")

    return results


EXAMPLE_IMAGENET_CLASSES = [
    "dog",
    "cat",
    "car",
    "airplane",
    "bird",
    "horse",
    "bicycle",
    "boat",
    "train",
    "truck",
    "person",
    "bottle",
    "chair",
    "dining_table",
    "potted_plant",
    "television",
    "laptop",
    "mouse",
    "keyboard",
    "cell_phone",
]

if __name__ == "__main__":
    import sys

    relevance_threshold = 3

    if len(sys.argv) > 1:
        class_names = [name.strip() for name in sys.argv[1].split(",")]
    else:
        class_names = EXAMPLE_IMAGENET_CLASSES

    synonyms_per_class = 7 if len(sys.argv) <= 2 else int(sys.argv[2])
    results = generate_synonyms_for_list(
        class_names, synonyms_per_class, relevance_threshold
    )
