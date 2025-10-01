import torch
from transformers import CLIPTextModel, CLIPTokenizer

device = "cuda" if torch.cuda.is_available() else "cpu"

# 1) Load tokenizer + text encoder
tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
clip_text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14").to(device).eval()

# 2) Some example texts
texts = [
    "A tabby cat riding a bicycle.",
    "A sunrise over a snowy mountain range."
]

# 3) Tokenize
enc = tokenizer(
    texts,
    padding=True,
    truncation=True,
    return_tensors="pt"
).to(device).input_ids

# 4) Encode (no grad)
with torch.no_grad():
    out = clip_text_encoder(**enc).pooler_output
    print(out)


print("Token embeddings:", last_hidden_state.shape)
print("Pooled text embeddings:", text_embeds.shape)
