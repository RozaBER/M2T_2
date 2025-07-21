#!/usr/bin/env python
"""
Quick test to verify training can start
"""
import os
import sys

# Set environment variables
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # Use single GPU for testing
os.environ['WANDB_MODE'] = 'offline'
os.environ['WANDB_SILENT'] = 'true'

# Temporarily reduce batch size for testing
import yaml

config_path = 'configs/default_config.yaml'
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

# Reduce batch size and steps for quick test
config['training']['batch_size'] = 4
config['training']['phase1_steps'] = 10
config['training']['phase2_steps'] = 5
config['training']['phase3_steps'] = 10
config['training']['logging_steps'] = 1
config['training']['eval_steps'] = 5
config['training']['save_steps'] = 10

# Save temporary config
test_config_path = 'configs/test_config.yaml'
with open(test_config_path, 'w') as f:
    yaml.dump(config, f)

print("Running quick training test...")
print(f"Batch size: {config['training']['batch_size']}")
print(f"Phase 1 steps: {config['training']['phase1_steps']}")

# Run training
import subprocess
cmd = [
    sys.executable, 'train.py',
    '--config', test_config_path,
    '--mode', 'multi_phase',
    '--output_dir', './output/test_run',
    '--device', 'cuda'
]

try:
    subprocess.run(cmd, check=True)
    print("\n✓ Training test completed successfully!")
except subprocess.CalledProcessError as e:
    print(f"\n✗ Training test failed with error code {e.returncode}")
    sys.exit(1)