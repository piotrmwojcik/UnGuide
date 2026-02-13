import os
from PIL import Image
import pandas as pd
import numpy as np
from transformers import CLIPProcessor, CLIPModel
from tqdm import tqdm
from argparse import ArgumentParser
import torch


def mean_clip_score(image_dir, prompts_path, max_images=10000):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = CLIPModel.from_pretrained(
        "openai/clip-vit-base-patch32",
        use_safetensors=True,
    ).eval().to(device)

    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    # Load prompts
    df = pd.read_csv(prompts_path)
    df.columns = df.columns.str.strip()

    case_to_prompt = {}

    for _, row in df.iterrows():
        try:
            case_number = int(row["case_number"])
        except (ValueError, TypeError):
            continue  # skip header or malformed rows

        case_to_prompt[str(case_number)] = row["prompt"]

    # Collect and sort images by numeric case id
    image_filenames = [
        f for f in os.listdir(image_dir)
        if f.endswith(".png")
    ]

    image_filenames = sorted(
        image_filenames,
        key=lambda x: int(os.path.splitext(x)[0])
    )

    similarities = []
    processed = 0

    for imagename in tqdm(image_filenames):
        if processed >= max_images:
            break

        case_id = str(int(os.path.splitext(imagename)[0]))

        if case_id not in case_to_prompt:
            continue

        text = case_to_prompt[case_id]

        image = Image.open(os.path.join(image_dir, imagename)).convert("RGB")

        inputs = processor(
            text=text,
            images=image,
            return_tensors="pt",
            padding=True,
        )

        outputs = model(**{k: v.to(device) for k, v in inputs.items()})

        clip_score = outputs.logits_per_image[0][0].detach().cpu()
        similarities.append(clip_score)

        processed += 1

    if processed != max_images:
        raise ValueError(f"Expected {max_images} images, but processed {processed}")

    similarities = np.array(similarities)

    mean_similarity = np.mean(similarities)
    std_similarity = np.std(similarities)

    print('-------------------------------------------------')
    print(f"Processed images: {processed}")
    print(f"Mean CLIP score ± Standard Deviation: {mean_similarity:.4f} ± {std_similarity:.4f}")


if __name__=='__main__':
    parser = ArgumentParser()
    parser.add_argument("--image_dir", type=str, default='path/to/generated_images')
    parser.add_argument("--prompts_path", type=str, default='./prompts_csv/coco_30k.csv')
    args = parser.parse_args()

    image_dir=args.image_dir
    prompts_path=args.prompts_path
    
    mean_clip_score(image_dir, prompts_path)
