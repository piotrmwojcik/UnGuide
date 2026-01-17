# pip install --upgrade sentence-transformers transformers torch matplotlib safetensors
import numpy as np
import matplotlib.pyplot as plt
import torch
from sentence_transformers import SentenceTransformer
from transformers import CLIPTextModel, CLIPTokenizer

# ----------------------------
# Models
# ----------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"

clip_name = "openai/clip-vit-base-patch32"
clip_tokenizer = CLIPTokenizer.from_pretrained(clip_name)
clip_text_model = CLIPTextModel.from_pretrained(
    clip_name,
    use_safetensors=True,
).to(device).eval()

sbert_name = "all-mpnet-base-v2"
sbert_model = SentenceTransformer(sbert_name)

# ----------------------------
# Labels
# ----------------------------
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

query = "dog"

# ----------------------------
# Helpers
# ----------------------------
@torch.no_grad()
def clip_text_embeddings(texts, batch_size=64):
    feats_all = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        enc = clip_tokenizer(batch, padding=True, truncation=True, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
        feats = clip_text_model(**enc).pooler_output
        feats = feats / feats.norm(dim=-1, keepdim=True)
        feats_all.append(feats.cpu())
    return torch.cat(feats_all, dim=0).numpy()

def pca_2d(E):
    X = E - E.mean(axis=0, keepdims=True)
    _, _, Vt = np.linalg.svd(X, full_matrices=False)
    return X @ Vt[:2].T

def save_pca(points2d, labels, title, filename, highlight="dog"):
    plt.figure(figsize=(10, 10))
    plt.scatter(points2d[:, 0], points2d[:, 1], s=20)

    highlight_set = {highlight, "cat", "wolf", "golden retriever", "corgie", "hound"}
    for i, lab in enumerate(labels):
        if lab in highlight_set:
            plt.annotate(lab, (points2d[i, 0], points2d[i, 1]),
                         fontsize=10, weight="bold")

    plt.title(title)
    plt.tight_layout()
    plt.savefig(filename, dpi=200)
    plt.close()
    print(f"Saved: {filename}")

# ----------------------------
# SBERT / MPNet PCA
# ----------------------------
texts_sem = cifar100 + [query]
E_sem = sbert_model.encode(texts_sem, normalize_embeddings=True)
X2_sem = pca_2d(E_sem)

save_pca(
    X2_sem,
    labels=texts_sem,
    title=f"SBERT ({sbert_name}) PCA – CIFAR objects + '{query}'",
    filename="pca_sbert_mpnet.png",
)

# ----------------------------
# CLIP PCA
# ----------------------------
clip_prompts = [f"A photo of a {c}" for c in cifar100] + [f"A photo of a {query}"]
E_clip = clip_text_embeddings(clip_prompts)
X2_clip = pca_2d(E_clip)

save_pca(
    X2_clip,
    labels=cifar100 + [query],
    title=f"CLIP ({clip_name}) PCA – CIFAR prompts + '{query}'",
    filename="pca_clip.png",
)
