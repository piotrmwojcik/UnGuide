# pip install --upgrade torch transformers
from transformers import CLIPTextModel, CLIPTokenizer
import torch
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"
model_name = "openai/clip-vit-base-patch32"

tokenizer = CLIPTokenizer.from_pretrained(model_name)
text_model = CLIPTextModel.from_pretrained(
    model_name,
    use_safetensors=True,   # require safetensors
).to(device).eval()

cifar100 = [
 'apple','aquarium fish','baby','bear','beaver','bed','bee','beetle','bicycle','bottle',
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

# A "lot" of dog synonyms / related terms (some are near-synonyms / hyponyms)
dog_synonyms = [
    "dog", "dogs", "domestic dog", "pet dog", "canine", "canid", "hound", "pooch", "puppy", "pup",
    "mutt", "mongrel", "cur", "stray dog", "street dog", "guard dog", "watchdog", "shepherd dog",
    "retriever", "terrier", "bulldog", "beagle", "labrador", "golden retriever", "german shepherd",
    "husky", "poodle", "chihuahua", "dachshund", "rottweiler", "doberman", "collie", "corgi",
    "shiba inu", "akita", "greyhound", "whippet", "spaniel", "boxer", "mastiff", "great dane",
    "saint bernard", "pit bull", "pug", "shih tzu", "schnauzer", "border collie", "cattle dog",
    "sled dog", "working dog"
]

@torch.no_grad()
def get_text_embeddings(texts, batch_size=64):
    all_feats = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        enc = tokenizer(batch, padding=True, truncation=True, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
        feats = text_model(**enc).pooler_output
        feats = feats / feats.norm(dim=-1, keepdim=True)
        all_feats.append(feats.detach().cpu())
    return torch.cat(all_feats, dim=0).numpy()

# CIFAR prompts
cifar_prompts = [f"A photo of a {c}" for c in cifar100]

# Multiple dog prompts (synonyms)
dog_prompts = [f"A photo of a {s}" for s in dog_synonyms]

# Embed everything
E_cifar = get_text_embeddings(cifar_prompts)     # [100, D]
E_dogs  = get_text_embeddings(dog_prompts)       # [S, D]

# Compute: for each dog synonym prompt, find nearest CIFAR classes
# We'll rank dog prompts by how close they get to the *best* CIFAR match.
best_per_syn = []
for si in range(E_dogs.shape[0]):
    sims = E_cifar @ E_dogs[si]          # [100]
    dists = 1.0 - sims
    best_idx = int(np.argmin(dists))
    best_dist = float(dists[best_idx])
    best_per_syn.append((best_dist, si, best_idx))

best_per_syn.sort(key=lambda x: x[0])

top_syn = 10  # show best 10 synonym prompts
print(f"Top {top_syn} dog-synonym prompts whose nearest CIFAR-100 class is closest (cosine distance):")
for rank, (best_dist, si, best_idx) in enumerate(best_per_syn[:top_syn], 1):
    print(f"{rank:2d}. '{dog_prompts[si]}'  ->  '{cifar100[best_idx]}'   dist={best_dist:.6f}")

# Also: using the single best synonym prompt, print top-5 CIFAR matches
best_dist, best_si, _ = best_per_syn[0]
sims = E_cifar @ E_dogs[best_si]
dists = 1.0 - sims
order = np.argsort(dists)

print("\nTop 5 CIFAR-100 classes closest to the BEST dog synonym prompt:")
print(f"Best synonym prompt: '{dog_prompts[best_si]}'")
for i in order[:5]:
    print(f"{cifar100[i]:15s}  cosine distance = {float(dists[i]):.6f}")
