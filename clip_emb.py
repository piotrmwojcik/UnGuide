# pip install --upgrade sentence-transformers
from sentence_transformers import SentenceTransformer
import numpy as np

# Load semantic embedding model
model_name = "all-MiniLM-L6-v2"
model = SentenceTransformer(model_name)

# Same CIFAR-100 list (plus cat)
cifar100 = [
 'apple', 'hound', 'golden retriever', 'corgie', 'aquarium fish','baby','bear','beaver','bed','bee','beetle','bicycle','bottle',
 'bowl','boy','bridge','bus','butterfly','camel','can','castle','caterpillar','cattle',
 'chair','chimpanzee','clock','cloud','cockroach','couch','crab','crocodile','cup','dinosaur',
 'dolphin','elephant','flatfish','forest','fox','girl','hamster','house','kangaroo','keyboard',
 'lamp','lawn mower','leopard','lion','lizard','lobster','man','maple tree','motorcycle','mountain',
 'mouse','mushroom','oak tree','orange','orchid','otter','palm tree','pear','pickup truck','pine tree',
 'plain','plate','poppy','porcupine','possum','rabbit','raccoon','ray','road','rocket',
 'rose','sea','seal','shark','shrew','skunk','skyscraper','snail','snake','spider',
 'squirrel','streetcar','sunflower','sweet pepper','table','tank','telephone','television','tiger','tractor',
 'train','trout','tulip','turtle','wardrobe','whale','willow tree','wolf','woman','worm', 'cat'
]

# Embed CIFAR words + query word "dog"
texts = cifar100 + ["dog"]

# Compute normalized semantic embeddings
E = model.encode(texts, normalize_embeddings=True)

dog_vec = E[-1]
others = E[:-1]

# Cosine distance
sims = others @ dog_vec
dists = 1.0 - sims
order = np.argsort(dists)

print("Top 10 CIFAR-100 classes closest to 'dog' (semantic embedding):")
for i in order[:10]:
    print(f"{cifar100[i]:15s}  cosine distance = {dists[i]:.6f}")
