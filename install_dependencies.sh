#!/bin/bash

echo "=== Installing Dependencies for Brain-to-Text Model ==="

# Determine Python command
if command -v python >/dev/null 2>&1; then
    PYTHON_CMD="python"
    PIP_CMD="pip"
elif command -v python3 >/dev/null 2>&1; then
    PYTHON_CMD="python3"
    PIP_CMD="pip3"
else
    echo "Error: Python not found. Please install Python 3.8+ first."
    exit 1
fi

echo "Using Python: $($PYTHON_CMD --version)"
echo "Using pip: $($PIP_CMD --version)"

# Upgrade pip
echo -e "\nUpgrading pip..."
$PIP_CMD install --upgrade pip

# Install PyTorch with CUDA support
echo -e "\nInstalling PyTorch with CUDA support..."
$PIP_CMD install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install other required packages
echo -e "\nInstalling other dependencies..."
$PIP_CMD install \
    transformers>=4.30.0 \
    numpy>=1.20.0 \
    pandas>=1.3.0 \
    mne>=1.0.0 \
    scikit-learn>=1.0.0 \
    wandb \
    tqdm \
    pyyaml \
    matplotlib \
    seaborn

echo -e "\n=== Installation Complete ==="
echo "You can now run the training with:"
echo "  ./setup_and_run.sh"
echo "or"
echo "  ./run_optimized_training.sh"