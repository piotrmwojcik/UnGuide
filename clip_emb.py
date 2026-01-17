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

# Candidate "synonyms"/near-synonyms for dog (you can add more)
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
    feats_all = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        enc = tokenizer(batch, padding=True, truncation=True, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
        feats = text_model(**enc).pooler_output
        feats = feats / feats.norm(dim=-1, keepdim=True)
        feats_all.append(feats.detach().cpu())
    return torch.cat(feats_all, dim=0).numpy()

# Embed the target prompt ("dog") and candidate synonyms
target_prompt = "A photo of a dog"
syn_prompts = [f"A photo of a {s}" for s in dog_synonyms]

E = get_text_embeddings([target_prompt] + syn_prompts)  # [1+S, D]
dog_vec = E[0]
syn_vecs = E[1:]

# Cosine distance to "dog" (vectors are normalized)
sims = syn_vecs @ dog_vec
dists = 1.0 - sims
order = np.argsort(dists)

topk = 10
print(f"Top {topk} 'synonyms' closest to '{target_prompt}' (cosine distance):")
for idx in order[:topk]:
    print(f"{dog_synonyms[idx]:20s}  dist={float(dists[idx]):.6f}")
