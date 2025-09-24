import random
import nltk
from nltk.corpus import wordnet as wn

def setup_wordnet():
    try:
        wn.synsets('test')
    except LookupError:
        nltk.download('wordnet', quiet=True)
        nltk.download('omw-1.4', quiet=True)

def get_random_object():
    synsets = list(wn.all_synsets(pos='n'))
    synset = random.choice(synsets)
    lemma = random.choice(synset.lemmas())
    return lemma.name().replace('_', ' ')

def generate_concepts(count=1):
    setup_wordnet()
    return [get_random_object() for _ in range(count)]

if __name__ == "__main__":
    import sys
    
    count = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    concepts = generate_concepts(count)
    
    for concept in concepts:
        print(concept)
