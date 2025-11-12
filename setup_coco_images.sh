#!/bin/bash
# Script to complete COCO image extraction and filter to keep only needed images

set -e

IMAGES_DIR="mscoco/images"
VAL2014_DIR="$IMAGES_DIR/val2014"
ZIP_FILE="$IMAGES_DIR/val2014.zip"
DOWNLOAD_URL="http://images.cocodataset.org/zips/val2014.zip"

echo "=== COCO Image Extraction and Filtering ==="
echo ""

# Create images directory if it doesn't exist
mkdir -p "$IMAGES_DIR"

# Download zip file if it doesn't exist
if [ ! -f "$ZIP_FILE" ]; then
    echo "Downloading COCO val2014 images..."
    echo "URL: $DOWNLOAD_URL"
    echo "Destination: $ZIP_FILE"
    wget -O "$ZIP_FILE" "$DOWNLOAD_URL"
    echo "Download completed!"
    echo ""
else
    echo "ZIP file already exists: $ZIP_FILE"
    echo ""
fi

# Check current image count
CURRENT_COUNT=$(ls -1 "$VAL2014_DIR" 2>/dev/null | wc -l)
echo "Current images in $VAL2014_DIR: $CURRENT_COUNT"

# Complete extraction if needed
if [ $CURRENT_COUNT -lt 40504 ]; then
    echo ""
    echo "Completing image extraction..."
    cd "$IMAGES_DIR"
    unzip -o -q val2014.zip
    cd -
    
    NEW_COUNT=$(ls -1 "$VAL2014_DIR" | wc -l)
    echo "Images after extraction: $NEW_COUNT"
else
    echo "All images already extracted."
fi

echo ""
echo "=== Filtering images based on CSV ==="
python filter_coco_images.py

echo ""
echo "=== Done ==="
