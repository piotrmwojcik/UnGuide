# pip install open_clip_torch torch torchvision matplotlib
import torch, numpy as np, matplotlib.pyplot as plt, open_clip

device = "cuda" if torch.cuda.is_available() else "cpu"
model, _, _ = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k', device=device)
tokenizer = open_clip.get_tokenizer('ViT-B-32')

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

prompts = [f"A photo of the {c}" for c in cifar100] + ["A photo of the cat"]

with torch.no_grad():
    toks = tokenizer(prompts).to(device)
    emb = model.encode_text(toks)
    emb = emb / emb.norm(dim=-1, keepdim=True)
    E = emb.float().cpu().numpy()

# PCA to 2D
X = E - E.mean(axis=0, keepdims=True)
U, S, Vt = np.linalg.svd(X, full_matrices=False)
X2 = X @ Vt[:2].T

plt.figure(figsize=(10,10))
plt.scatter(X2[:-1,0], X2[:-1,1], s=20, label="CIFAR-100 prompts")
plt.scatter(X2[-1,0], X2[-1,1], s=80, marker='*', label="A photo of a cat")
# annotate a few random labels
import numpy as np; rng = np.random.default_rng(0)
for i in rng.choice(len(cifar100), size=20, replace=False):
    plt.annotate(cifar100[i], (X2[i,0], X2[i,1]), fontsize=8)
plt.annotate("cat", (X2[-1,0], X2[-1,1]), fontsize=10, weight='bold')
plt.legend()
plt.title("CLIP text embeddings: CIFAR-100 prompts + 'A photo of a cat' (PCA)")
plt.tight_layout()
plt.savefig("cifar100_clip_text_prompts_plot.png", dpi=200)
print("Saved plot to cifar100_clip_text_prompts_plot.png")
