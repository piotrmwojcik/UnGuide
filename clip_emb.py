# pip install transformers torch matplotlib --upgrade
from transformers import CLIPTextModel, CLIPTokenizer
import torch, numpy as np, matplotlib.pyplot as plt

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load text tower (uses pooled EOT embedding; projection from CLIPModel is not used here)
model_name = "openai/clip-vit-base-patch32"
tokenizer = CLIPTokenizer.from_pretrained(model_name)
text_model = CLIPTextModel.from_pretrained(model_name).to(device).eval()

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
    'train','trout','tulip','turtle','wardrobe','whale','willow tree','wolf','woman','worm'
]

prompts = [f"A photo of a {c}" for c in cifar100] + ["A photo of a cat"]

@torch.no_grad()
def get_text_embeddings(texts):
    # Tokenize
    enc = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
    enc = {k: v.to(device) for k, v in enc.items()}
    # Forward through CLIPTextModel
    out = text_model(**enc)
    # Use pooled EOT embedding (out.pooler_output) as the sentence embedding
    feats = out.pooler_output  # [N, hidden_size]
    feats = feats / feats.norm(dim=-1, keepdim=True)  # L2-normalize
    return feats.cpu().numpy()

E = get_text_embeddings(prompts)

# PCA (via SVD) to 2D for plotting
X = E - E.mean(axis=0, keepdims=True)
U, S, Vt = np.linalg.svd(X, full_matrices=False)
X2 = X @ Vt[:2].T  # [N, 2]

# Plot and save
plt.figure(figsize=(10, 10))
plt.scatter(X2[:-1, 0], X2[:-1, 1], s=20, label="CIFAR-100 prompts")
plt.scatter(X2[-1, 0], X2[-1, 1], s=80, marker='*', label="A photo of a cat")
# annotate a few points to avoid clutter
rng = np.random.default_rng(0)
for i in rng.choice(len(cifar100), size=20, replace=False):
    plt.annotate(cifar100[i], (X2[i, 0], X2[i, 1]), fontsize=8)
plt.annotate("cat", (X2[-1, 0], X2[-1, 1]), fontsize=10, weight='bold')
plt.legend()
plt.title("CLIP text embeddings: CIFAR-100 prompts + 'A photo of a cat' (PCA)")
plt.tight_layout()
plt.savefig("cifar100_clip_text_prompts_plot.png", dpi=200)
print("Saved to cifar100_clip_text_prompts_plot.png")
