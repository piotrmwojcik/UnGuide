import torch
from sentence_transformers import SentenceTransformer
import pandas as pd
from tqdm import tqdm
import os
import argparse

def load_model(model_name='intfloat/e5-large-v2', models_dir='models', device='cuda'):
    local_path = os.path.join(models_dir, model_name.split('/')[-1])
    
    if os.path.exists(local_path):
        model = SentenceTransformer(local_path, device=device)
    else:
        os.makedirs(models_dir, exist_ok=True)
        model = SentenceTransformer(model_name, device=device)
        model.save(local_path)
    
    return model

def load_prompts(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        prompts = [line.strip() for line in f if line.strip()]
    return prompts

def encode_batch(model, texts, batch_size=64):
    all_embeddings = []
    
    for i in tqdm(range(0, len(texts), batch_size), desc="Encoding"):
        batch = texts[i:i + batch_size]
        batch_with_prefix = [f"query: {text}" for text in batch]
        
        with torch.no_grad():
            embeddings = model.encode(
                batch_with_prefix,
                batch_size=batch_size,
                show_progress_bar=False,
                convert_to_tensor=True,
                normalize_embeddings=True
            )
        
        all_embeddings.append(embeddings.cpu())
        torch.cuda.empty_cache()
    
    return torch.cat(all_embeddings, dim=0)

def save_to_csv(prompts, embeddings, output_path):
    embeddings_np = embeddings.numpy()
    embeddings_list = [emb.tolist() for emb in embeddings_np]
    
    df = pd.DataFrame({
        'prompt': prompts,
        'embedding': embeddings_list
    })
    df.to_csv(output_path, index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Encode prompts using E5-large-v2')
    parser.add_argument('--input_file', default='concepts.txt', help='Path to input text file with concepts')
    parser.add_argument('--output_file', default='embeddings.csv', help='Path to output CSV file')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for encoding')
    
    args = parser.parse_args()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = load_model(device=device)
    
    prompts = load_prompts(args.input_file)
    embeddings = encode_batch(model, prompts, args.batch_size)
    save_to_csv(prompts, embeddings, args.output_file)
