#!/usr/bin/env python
"""
Direct Python script to run training without shell complications
"""
import os
import sys
import subprocess

# Set environment variables
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
os.environ['NCCL_DEBUG'] = 'INFO'
os.environ['TORCH_DISTRIBUTED_DEBUG'] = 'DETAIL'
os.environ['WANDB_MODE'] = 'offline'  # Use offline mode by default
os.environ['WANDB_SILENT'] = 'true'  # Silence wandb warnings

# Create directories
os.makedirs('./output/optimized_run', exist_ok=True)
os.makedirs('./cache', exist_ok=True)
os.makedirs('./wandb', exist_ok=True)

print("=== Brain-to-Text Optimized Training ===")
print(f"Python: {sys.version}")
print(f"Python executable: {sys.executable}")

# Check packages
required_packages = ['torch', 'transformers', 'numpy', 'pandas', 'mne', 'sklearn', 'wandb', 'tqdm', 'yaml']
missing_packages = []

for package in required_packages:
    try:
        if package == 'sklearn':
            __import__('sklearn')
        elif package == 'yaml':
            __import__('yaml')
        else:
            __import__(package)
        print(f"✓ {package} is installed")
    except ImportError:
        print(f"✗ {package} is NOT installed")
        missing_packages.append(package)

if missing_packages:
    print(f"\nError: Missing packages: {missing_packages}")
    print("Please install them with:")
    print(f"pip install {' '.join(missing_packages)}")
    sys.exit(1)

# Check CUDA
try:
    import torch
    if torch.cuda.is_available():
        print(f"\n✓ CUDA is available")
        print(f"  - Number of GPUs: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  - GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        print("\n⚠ Warning: CUDA not available, will use CPU")
except Exception as e:
    print(f"\n⚠ Warning: Could not check CUDA: {e}")

# Run training
print("\n=== Starting Training ===")
cmd = [
    sys.executable, 'train.py',
    '--config', 'configs/default_config.yaml',
    '--mode', 'multi_phase',
    '--output_dir', './output/optimized_run',
    '--device', 'cuda',
    '--wandb'
]

print(f"Command: {' '.join(cmd)}")
print("\nTraining output:")
print("-" * 50)

# Run the training
try:
    subprocess.run(cmd, check=True)
    print("\n=== Training Completed Successfully ===")
except subprocess.CalledProcessError as e:
    print(f"\n=== Training Failed with exit code {e.returncode} ===")
    sys.exit(1)
except KeyboardInterrupt:
    print("\n=== Training Interrupted by User ===")
    sys.exit(0)