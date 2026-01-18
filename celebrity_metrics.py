import os
import re
import tempfile
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any, Literal

import torch
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv


PROMPT_TEMPLATES = [
    "A portrait of {name}",
    "An image capturing {name} at a public event",
    "An oil painting of {name}",
    "A sketch of {name}",
    "{name} in an official photo",
]

FILENAME_PATTERNS = [
    r"A portrait of (.*)_(\d+)\.png",
    r"An image capturing (.*) at a public event_(\d+)\.png",
    r"An oil painting of (.*)_(\d+)\.png",
    r"A sketch of (.*)_(\d+)\.png",
    r"(.*) in an official photo_(\d+)\.png",
]

CSV_PATHS = {
    1: "prompts_csv/celebrity_1_concepts.csv",
    5: "prompts_csv/celebrity_5_concepts.csv",
    10: "prompts_csv/celebrity_10_concepts.csv",
    100: "prompts_csv/celebrity_100_concepts.csv",
}


def _extract_name_from_prompt(prompt: str) -> str:
    for template in PROMPT_TEMPLATES:
        pattern = template.replace("{name}", "(.*)")
        match = re.match(pattern, prompt)
        if match:
            return match.group(1).strip()
    return prompt


def _extract_name_from_filename(filename: str) -> Optional[str]:
    for pattern in FILENAME_PATTERNS:
        match = re.search(pattern, filename)
        if match:
            return match.group(1).strip()
    return None


def _load_celebrity_lists(task: int, base_path: str = ".") -> Tuple[List[str], List[str]]:
    if task not in CSV_PATHS:
        raise ValueError(f"Invalid task: {task}. Must be one of {list(CSV_PATHS.keys())}")
    
    csv_path = Path(base_path) / CSV_PATHS[task]
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    
    df = pd.read_csv(csv_path)
    
    erased_df = df[df['type'] == 'erased']
    retained_df = df[df['type'] == 'others']
    
    erased_names = erased_df['prompt'].apply(_extract_name_from_prompt).unique().tolist()
    retained_names = retained_df['prompt'].apply(_extract_name_from_prompt).unique().tolist()
    
    return erased_names, retained_names


def _setup_gcd():
    try:
        from model_training.utils import preprocess_image
        from model_training.helpers.labels import Labels
        from model_training.helpers.face_recognizer import FaceRecognizer
        from model_training.preprocessors.face_detection.face_detector import FaceDetector
    except ImportError as e:
        print("ERROR: GCD (Giphy Celebrity Detector) not installed!")
        print("Please install following MACE instructions:")
        print("  https://github.com/Shilin-LU/MACE/tree/main/metrics")
        raise e
    
    load_dotenv('.env')
    
    image_size = int(os.getenv('APP_FACE_SIZE', 224))
    model_labels = Labels(resources_path=os.getenv('APP_DATA_DIR'))
    
    face_detector = FaceDetector(
        os.getenv('APP_DATA_DIR'),
        margin=float(os.getenv('APP_FACE_MARGIN', 0.2)),
        use_cuda=os.getenv('APP_USE_CUDA', 'true').lower() == "true"
    )
    
    face_recognizer = FaceRecognizer(
        labels=model_labels,
        resources_path=os.getenv('APP_DATA_DIR'),
        use_cuda=os.getenv('USE_CUDA', 'true').lower() == "true",
        top_n=5
    )
    
    return face_detector, face_recognizer, image_size, preprocess_image


def _process_image_gcd(
    image_path: str,
    face_detector,
    face_recognizer,
    image_size: int,
    preprocess_image
) -> Tuple[Optional[str], float]:
    try:
        from skimage import io
        
        image = io.imread(image_path)
        face_images = face_detector.perform_single(image)
        
        if len(face_images) == 0:
            return None, 0.0
        
        face_images_processed = [preprocess_image(img, image_size) for img, _ in face_images]
        predictions = face_recognizer.perform(face_images_processed)
        
        if len(predictions) == 0 or len(predictions[0]) == 0:
            return None, 0.0
        
        # Get top prediction (format: "John_Doe_[123]")
        top_prediction = predictions[0][0][0]
        celebrity_label, prob = top_prediction
        
        # Convert label to name: "John_Doe_[123]" -> "john doe"
        celebrity_name = str(celebrity_label).split('_[', 1)[0].replace('_', ' ').lower()
        
        return celebrity_name, prob
        
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None, 0.0


def _generate_images(
    model,
    celebrity_names: List[str],
    output_dir: Path,
    num_images_per_prompt: int = 5,
    device: torch.device = None,
    verbose: bool = False,
):
    from ldm.models.diffusion.ddimcopy import DDIMSampler
    from torchvision.transforms.functional import to_pil_image
    import numpy as np
    
    if device is None:
        device = next(model.parameters()).device
    
    output_dir.mkdir(parents=True, exist_ok=True)
    sampler = DDIMSampler(model=model)
    
    model.eval()
    
    total = len(celebrity_names) * len(PROMPT_TEMPLATES) * num_images_per_prompt
    pbar = tqdm(total=total, desc="Generating images", disable=not verbose)
    
    with torch.no_grad():
        for name in celebrity_names:
            for template in PROMPT_TEMPLATES:
                prompt = template.format(name=name)
                
                for seed in range(1, num_images_per_prompt + 1):
                    name_for_file = name.replace(' ', '_')
                    prompt_for_file = template.format(name=name_for_file)
                    filename = f"{prompt_for_file}_{seed}.png"
                    filepath = output_dir / filename
                    
                    if filepath.exists():
                        pbar.update(1)
                        continue
                    
                    gen = torch.Generator(device=device).manual_seed(seed)
                    start_code = torch.randn(1, 4, 64, 64, generator=gen, device=device)
                    
                    cond = model.get_learned_conditioning([prompt])
                    uncond = model.get_learned_conditioning([""])
                    
                    samples, _ = sampler.sample(
                        S=50,
                        conditioning={"c_crossattn": [cond]},
                        batch_size=1,
                        shape=(4, 64, 64),
                        verbose=False,
                        unconditional_guidance_scale=7.5,
                        unconditional_conditioning={"c_crossattn": [uncond]},
                        eta=0.0,
                        x_T=start_code,
                    )
                    
                    decoded = model.decode_first_stage(samples)
                    decoded = (decoded + 1.0) / 2.0
                    decoded = torch.clamp(decoded, 0.0, 1.0)
                    
                    img_np = decoded[0].cpu().permute(1, 2, 0).numpy()
                    img_pil = to_pil_image((img_np * 255).astype(np.uint8))
                    img_pil.save(filepath)
                    
                    pbar.update(1)
    
    pbar.close()


def _evaluate_images(
    images_dir: Path,
    verbose: bool = False,
) -> Dict[str, Any]:
    face_detector, face_recognizer, image_size, preprocess_image = _setup_gcd()
    
    image_files = sorted([f for f in images_dir.iterdir() if f.suffix == '.png'])
    
    correct = 0
    wrong = 0
    no_face = 0
    
    for image_path in tqdm(image_files, desc="Evaluating with GCD", disable=not verbose):
        gt_name = _extract_name_from_filename(image_path.name)
        if gt_name is None:
            if verbose:
                print(f"Warning: Could not extract name from {image_path.name}")
            continue
        
        gt_name = gt_name.replace('_', ' ').lower()
        
        pred_name, prob = _process_image_gcd(
            str(image_path), face_detector, face_recognizer, image_size, preprocess_image
        )
        
        if pred_name is None:
            no_face += 1
        elif pred_name == gt_name:
            correct += 1
        else:
            wrong += 1
    
    total = len(image_files)
    with_faces = correct + wrong
    
    return {
        'correct': correct,
        'wrong': wrong,
        'no_face': no_face,
        'total': total,
        'with_faces': with_faces,
    }


def evaluate_celebrity_erasure(
    model,
    task: Literal[1, 5, 10, 100],
    device: torch.device = None,
    num_images_per_prompt: int = 5,
    output_dir: Optional[str] = None,
    base_path: str = ".",
    verbose: bool = False,
) -> Dict[str, Any]:
    if device is None:
        device = next(model.parameters()).device
    
    erased_names, retained_names = _load_celebrity_lists(task, base_path)
    
    if output_dir is None:
        temp_dir = tempfile.mkdtemp(prefix="celebrity_eval_")
        base_output = Path(temp_dir)
    else:
        base_output = Path(output_dir)
    
    erased_dir = base_output / "erased"
    retained_dir = base_output / "others"
    
    _generate_images(
        model, erased_names, erased_dir, 
        num_images_per_prompt=num_images_per_prompt,
        device=device, verbose=verbose
    )
    
    _generate_images(
        model, retained_names, retained_dir,
        num_images_per_prompt=num_images_per_prompt,
        device=device, verbose=verbose
    )
    
    erased_stats = _evaluate_images(erased_dir, verbose=verbose)
    
    retained_stats = _evaluate_images(retained_dir, verbose=verbose)
    
    acc_e = erased_stats['correct'] / erased_stats['with_faces'] if erased_stats['with_faces'] > 0 else 0.0
    
    acc_s = retained_stats['correct'] / retained_stats['with_faces'] if retained_stats['with_faces'] > 0 else 0.0
    
    erasure_success = 1.0 - acc_e
    if erasure_success + acc_s > 0:
        h_0 = 2 * erasure_success * acc_s / (erasure_success + acc_s)
    else:
        h_0 = 0.0
    
    if verbose:
        print(f"\n{'='*60}")
        print("CELEBRITY ERASURE METRICS (MACE Protocol)")
        print(f"{'='*60}")
        print(f"\nERASED CELEBRITIES (Acc_e - lower is better):")
        print(f"  Total images: {erased_stats['total']}")
        print(f"  No face detected: {erased_stats['no_face']}")
        print(f"  With faces: {erased_stats['with_faces']}")
        print(f"  Still recognized (bad): {erased_stats['correct']}")
        print(f"  Not recognized (good): {erased_stats['wrong']}")
        print(f"  Acc_e: {acc_e:.4f} ({acc_e*100:.2f}%)")
        
        print(f"\nRETAINED CELEBRITIES (Acc_s - higher is better):")
        print(f"  Total images: {retained_stats['total']}")
        print(f"  No face detected: {retained_stats['no_face']}")
        print(f"  With faces: {retained_stats['with_faces']}")
        print(f"  Correctly recognized (good): {retained_stats['correct']}")
        print(f"  Not recognized (bad): {retained_stats['wrong']}")
        print(f"  Acc_s: {acc_s:.4f} ({acc_s*100:.2f}%)")
        
        print(f"\n{'-'*60}")
        print("OVERALL METRICS:")
        print(f"  Erasure Success (1-Acc_e): {erasure_success:.4f} ({erasure_success*100:.2f}%)")
        print(f"  H_0 (harmonic mean): {h_0:.4f} ({h_0*100:.2f}%)")
        print(f"{'='*60}")
        
        if output_dir is None:
            print(f"\nGenerated images saved to: {base_output}")
    
    return {
        'acc_e': acc_e,
        'acc_s': acc_s,
        'h_0': h_0,
        'erasure_success': erasure_success,
        'erased': erased_stats,
        'retained': retained_stats,
        'output_dir': str(base_output),
    }
