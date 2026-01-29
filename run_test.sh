#!/bin/bash
# UnGuide Test Script
# Creates environment, installs dependencies, runs training and generation tests

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "========================================"
echo "UnGuide Test Suite"
echo "========================================"

# Check Python version (requires 3.8+)
PYTHON_CMD="${PYTHON:-python3}"

# Try to find a suitable Python
for cmd in python3.10 python3.9 python3.8 python3; do
    if command -v $cmd &> /dev/null; then
        version=$($cmd -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
        major=$(echo $version | cut -d. -f1)
        minor=$(echo $version | cut -d. -f2)
        if [ "$major" -ge 3 ] && [ "$minor" -ge 8 ]; then
            PYTHON_CMD=$cmd
            break
        fi
    fi
done

echo "Using Python: $PYTHON_CMD"
$PYTHON_CMD --version

# Verify Python version
$PYTHON_CMD -c "import sys; assert sys.version_info >= (3, 8), 'Python 3.8+ required'" || {
    echo "ERROR: Python 3.8+ is required. Found: $($PYTHON_CMD --version)"
    echo "Please install Python 3.8+ or set PYTHON environment variable."
    echo "Example: PYTHON=/path/to/python3.10 ./run_test.sh"
    exit 1
}

# Create virtual environment
echo ""
echo "[1/7] Creating virtual environment: unhype_test"
if [ ! -d "unhype_test" ]; then
    $PYTHON_CMD -m venv unhype_test
fi
source unhype_test/bin/activate

# Install requirements
echo ""
echo "[2/7] Installing requirements..."
pip install --upgrade pip
pip install -r requirements.txt

# Create output directory
mkdir -p output

# Check if SD model exists
if [ ! -f "models/sd-v1-4.ckpt" ]; then
    echo ""
    echo "WARNING: models/sd-v1-4.ckpt not found!"
    echo "Please download Stable Diffusion v1.4 checkpoint to models/sd-v1-4.ckpt"
    echo "You can download it from: https://huggingface.co/CompVis/stable-diffusion-v-1-4-original"
    echo ""
fi

# Training 1: CIFAR-10 Airplane
echo ""
echo "[3/7] Training: CIFAR-10 Airplane..."
python train_sd.py --config configs_test/cifar_10/train_airplane.yaml || {
    echo "Training failed for cifar10_airplane"
}

# Training 2: Celebrity CLIP
echo ""
echo "[4/7] Training: Celebrity CLIP..."
python train_sd.py --config configs_test/celebrity/train_celebrity_clip.yaml || {
    echo "Training failed for celebrity_clip"
}

# Training 3: Celebrity NV-Embed
echo ""
echo "[5/7] Training: Celebrity NV-Embed..."
python train_sd.py --config configs_test/celebrity/train_celebrity_nvembed.yaml || {
    echo "Training failed for celebrity_nvembed"
}

# Training 4: Nudity SD
echo ""
echo "[6/7] Training: Nudity SD..."
python train_sd.py --config configs_test/nudity/nudity_sd.yaml || {
    echo "Training failed for nudity_sd"
}

# Training 5: Nudity Flux
echo ""
echo "[7/7] Training: Nudity Flux..."
python train_flux.py --config configs_test/nudity/nudity_flux.yaml || {
    echo "Training failed for nudity_flux"
}

echo ""
echo "========================================"
echo "Training Complete. Generating Images..."
echo "========================================"

# Create test prompts CSV
cat > output/test_prompts.csv << 'EOF'
,prompt,evaluation_seed
0,a photo of adam driver,42
EOF

# Generate from each model
echo ""
echo "Generating from CIFAR-10 Airplane model..."
python generate_sd.py \
    --task cifar10 \
    --config configs_test/cifar_10/train_airplane.yaml \
    --lora-path output/test_cifar10_airplane/LoRA_fusion_model \
    --output-dir output/test_cifar10_airplane/images \
    --n-images 1 \
    --samples-per-prompt 1 || echo "Generation failed for cifar10"

echo ""
echo "Generating from Celebrity CLIP model..."
python generate_sd.py \
    --task celebrity \
    --config configs_test/celebrity/train_celebrity_clip.yaml \
    --lora-path output/test_celebrity_clip/LoRA_fusion_model \
    --prompts-csv output/test_prompts.csv \
    --output-dir output/test_celebrity_clip/images \
    --n-images 1 || echo "Generation failed for celebrity_clip"

echo ""
echo "Generating from Celebrity NV-Embed model..."
python generate_sd.py \
    --task celebrity \
    --config configs_test/celebrity/train_celebrity_nvembed.yaml \
    --lora-path output/test_celebrity_nvembed/LoRA_fusion_model \
    --prompts-csv output/test_prompts.csv \
    --output-dir output/test_celebrity_nvembed/images \
    --n-images 1 || echo "Generation failed for celebrity_nvembed"

echo ""
echo "Generating from Nudity SD model..."
python generate_sd.py \
    --task nudity \
    --config configs_test/nudity/nudity_sd.yaml \
    --lora-path output/test_nudity_sd/LoRA_fusion_model \
    --prompts-csv output/test_prompts.csv \
    --output-dir output/test_nudity_sd/images \
    --n-images 1 || echo "Generation failed for nudity_sd"

echo ""
echo "Generating from Nudity Flux model..."
python generate_flux.py \
    --csv_path output/test_prompts.csv \
    --lora_path output/test_nudity_flux/LoRA_model \
    --output_dir output/test_nudity_flux/images \
    --n_images 1 || echo "Generation failed for nudity_flux"

echo ""
echo "========================================"
echo "Test Complete!"
echo "========================================"
echo ""
echo "Results saved to:"
echo "  - output/test_cifar10_airplane/"
echo "  - output/test_celebrity_clip/"
echo "  - output/test_celebrity_nvembed/"
echo "  - output/test_nudity_sd/"
echo "  - output/test_nudity_flux/"
echo ""

# List generated images
echo "Generated images:"
find output/test_*/images -name "*.png" -o -name "*.jpg" 2>/dev/null || echo "No images found"
