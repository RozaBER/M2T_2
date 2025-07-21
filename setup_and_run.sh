#!/bin/bash

echo "=== Brain-to-Text Optimized Training Setup ==="

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check Python availability
echo "Checking Python installation..."
if command_exists python; then
    PYTHON_CMD="python"
elif command_exists python3; then
    PYTHON_CMD="python3"
else
    echo "Error: Python not found. Please install Python 3.8+ or activate your environment."
    exit 1
fi

echo "Using Python: $($PYTHON_CMD --version)"

# Check CUDA availability
echo -e "\nChecking CUDA availability..."
if command_exists nvidia-smi; then
    nvidia-smi --query-gpu=name,memory.total --format=csv
else
    echo "Warning: nvidia-smi not found. GPU might not be available."
fi

# Check required packages
echo -e "\nChecking required packages..."
$PYTHON_CMD -c "
import sys
print(f'Python version: {sys.version}')

required_packages = [
    'torch',
    'transformers',
    'numpy',
    'pandas',
    'mne',
    'sklearn',
    'wandb',
    'tqdm',
    'yaml'
]

missing_packages = []
for package in required_packages:
    try:
        if package == 'yaml':
            import yaml
        else:
            __import__(package)
        print(f'✓ {package} is installed')
    except ImportError:
        print(f'✗ {package} is NOT installed')
        missing_packages.append(package)

if missing_packages:
    print(f'\nMissing packages: {missing_packages}')
    print('Please install missing packages with:')
    print(f'pip install {\" \".join(missing_packages)}')
    sys.exit(1)
else:
    print('\n✓ All required packages are installed')
"

if [ $? -ne 0 ]; then
    echo -e "\nError: Missing required packages. Please install them first."
    echo "You can use: pip install torch transformers numpy pandas mne scikit-learn wandb tqdm pyyaml"
    exit 1
fi

# Create necessary directories
echo -e "\nCreating output directories..."
mkdir -p ./output/optimized_run
mkdir -p ./cache
mkdir -p ./wandb

# Set environment variables for multi-GPU training
export CUDA_VISIBLE_DEVICES=0,1
export NCCL_DEBUG=INFO
export TORCH_DISTRIBUTED_DEBUG=DETAIL

# Optional: Set wandb environment variables
# export WANDB_API_KEY="your_api_key_here"
# export WANDB_PROJECT="brain-to-text-optimized"

# Check if wandb is configured, otherwise use offline mode
if [ -z "$WANDB_API_KEY" ] && ! [ -f ~/.netrc ] || ! grep -q "api.wandb.ai" ~/.netrc 2>/dev/null; then
    echo "wandb not configured, using offline mode"
    export WANDB_MODE=offline
fi

echo -e "\n=== Starting Optimized Training ==="
echo "Configuration:"
echo "  - GPUs: $CUDA_VISIBLE_DEVICES"
echo "  - Config: configs/default_config.yaml"
echo "  - Output: ./output/optimized_run"
echo "  - Mode: multi_phase"

# Run training
$PYTHON_CMD train.py \
    --config configs/default_config.yaml \
    --mode multi_phase \
    --output_dir ./output/optimized_run \
    --device cuda \
    --wandb \
    2>&1 | tee ./output/optimized_run/training.log

echo -e "\n=== Training Completed ==="
echo "Results saved to: ./output/optimized_run"
echo "Training log: ./output/optimized_run/training.log"
echo "Checkpoints: ./output/optimized_run/Phase_*/"