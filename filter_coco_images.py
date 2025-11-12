#!/usr/bin/env python3
"""
Filter COCO validation images to keep only those referenced in the CSV file.
"""

import os
import pandas as pd
from pathlib import Path
from tqdm import tqdm

def main():
    # Paths
    csv_path = "mscoco/coco_30k.csv"
    images_dir = Path("mscoco/images/val2014")
    
    # Read CSV to get the list of COCO IDs we need
    print(f"Reading CSV file: {csv_path}")
    df = pd.read_csv(csv_path)
    needed_ids = set(df['coco_id'].astype(str))
    print(f"Found {len(needed_ids)} unique COCO IDs in CSV")
    
    # Get all image files in the directory
    all_images = list(images_dir.glob("COCO_val2014_*.jpg"))
    print(f"Found {len(all_images)} total images in {images_dir}")
    
    # Track images to keep and remove
    to_keep = []
    to_remove = []
    
    for img_path in tqdm(all_images, desc="Processing images"):
        # Extract COCO ID from filename
        # Format: COCO_val2014_000000123456.jpg
        filename = img_path.name
        coco_id = filename.replace("COCO_val2014_", "").replace(".jpg", "").lstrip("0") or "0"
        
        if coco_id in needed_ids:
            to_keep.append(img_path)
        else:
            to_remove.append(img_path)
    
    print(f"\nImages to keep: {len(to_keep)}")
    print(f"Images to remove: {len(to_remove)}")
    
    # Ask for confirmation
    if to_remove:
        response = input(f"\nDo you want to delete {len(to_remove)} images? (yes/no): ")
        if response.lower() == 'yes':
            print("\nDeleting unnecessary images...")
            for img_path in tqdm(to_remove, desc="Deleting"):
                img_path.unlink()
            print(f"✓ Successfully deleted {len(to_remove)} images")
            print(f"✓ Kept {len(to_keep)} images referenced in CSV")
        else:
            print("Deletion cancelled.")
    else:
        print("\nNo images to remove. All images are referenced in the CSV!")

if __name__ == "__main__":
    main()
