#!/usr/bin/env python
"""Debug training issues"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['WANDB_MODE'] = 'offline'
os.environ['WANDB_SILENT'] = 'true'

import torch
import sys
sys.path.append('.')

print("1. Testing imports...")
try:
    from src.models.full_model import BrainToTextModel
    from src.data.meg_dataset import MEGDataset, MEGDataCollator
    from src.training.multi_phase_trainer_optimized import OptimizedMultiPhaseTrainer
    print("✓ Imports successful")
except Exception as e:
    print(f"✗ Import error: {e}")
    sys.exit(1)

print("\n2. Testing model creation...")
try:
    model = BrainToTextModel(
        n_channels=208,
        eeg_d_model=768,
        eeg_n_layers=12,
        num_quantizers=8,
        codebook_size=8192,
        vocab_size=32000,
        freeze_llm=True
    )
    print(f"✓ Model created: {model.get_num_params():,} params")
except Exception as e:
    print(f"✗ Model creation error: {e}")
    sys.exit(1)

print("\n3. Testing forward pass...")
try:
    # Create dummy input
    batch_size = 2
    channels = 208
    time_steps = 2000
    
    dummy_input = {
        'eeg_signals': torch.randn(batch_size, channels, time_steps).cuda(),
        'input_ids': torch.randint(0, 32000, (batch_size, 10)).cuda(),
        'attention_mask': torch.ones(batch_size, 10).cuda(),
        'labels': torch.randint(0, 32000, (batch_size, 10)).cuda()
    }
    
    model = model.cuda()
    model.train()
    
    # Test forward pass
    with torch.cuda.amp.autocast(enabled=True):
        outputs = model(**dummy_input)
    
    print(f"✓ Forward pass successful")
    print(f"  - Loss: {outputs['loss'].item():.4f}")
    print(f"  - Output keys: {list(outputs.keys())}")
    
except Exception as e:
    print(f"✗ Forward pass error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n4. Testing RVQ module...")
try:
    # Test RVQ stats
    rvq_stats = model.rvq.get_codebook_usage()
    print(f"✓ RVQ stats retrieved")
    print(f"  - Temperature: {rvq_stats.get('overall', {}).get('temperature', 'N/A')}")
    
except Exception as e:
    print(f"✗ RVQ error: {e}")

print("\n✓ All tests passed! Training should work.")
print("\nTo run full training:")
print("  python train.py --config configs/default_config.yaml --mode multi_phase --output_dir ./output/optimized_run --device cuda")