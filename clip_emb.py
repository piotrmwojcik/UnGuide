# pip install --upgrade torch transformers
from transformers import CLIPTextModel, CLIPTokenizer
import torch
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"
model_name = "openai/clip-vit-base-patch32"

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
 'train','trout','tulip','turtle','wardrobe','whale','willow tree','wolf','woman','worm'
]

prompts = [f"A photo of a {c}" for c in cifar100] + ["A photo of a dog"]

@torch.no_grad()
def get_text_embeddings(texts):
    enc = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
    enc = {k: v.to(device) for k, v in enc.items()}
    feats = text_model(**enc).pooler_output
    feats = feats / feats.norm(dim=-1, keepdim=True)
    return feats.cpu().numpy()

E = get_text_embeddings(prompts)

dog_vec = E[-1]
others = E[:-1]

sims = others @ dog_vec
dists = 1.0 - sims
order = np.argsort(dists)

print("Top 5 CIFAR-100 classes closest to 'A photo of a dog':")
for i in order[:5]:
    print(f"{cifar100[i]:15s}  cosine distance = {dists[i]:.6f}")
