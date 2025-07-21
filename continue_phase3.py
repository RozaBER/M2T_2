#!/usr/bin/env python3
"""
Continue training from Phase 3 after Phase 2 completion
"""

import argparse
import torch
import os
from pathlib import Path
import json

from train import load_config, setup_dataset
from src.models.full_model import BrainToTextModel
from src.training.multi_phase_trainer import MultiPhaseTrainer


def save_phase2_checkpoint(model, output_dir):
    """Save the current model state as phase2_complete"""
    phase2_dir = Path(output_dir) / "phase2_complete"
    phase2_dir.mkdir(parents=True, exist_ok=True)
    
    # Save model
    model.save_pretrained(str(phase2_dir))
    
    # Save phase info
    phase_info = {
        "phase": 2,
        "phase_name": "phase2_complete",
        "model_state": {
            "encoder_frozen": True,
            "rvq_frozen": False,
            "llm_frozen": True
        }
    }
    
    with open(phase2_dir / "phase_info.json", "w") as f:
        json.dump(phase_info, f, indent=2)
    
    print(f"Saved phase 2 checkpoint to {phase2_dir}")
    return phase2_dir


def main():
    parser = argparse.ArgumentParser(description='Continue training from Phase 3')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to configuration file')
    parser.add_argument('--phase1_checkpoint', type=str, 
                       default='./output/phase1_complete',
                       help='Path to phase 1 checkpoint directory')
    parser.add_argument('--output_dir', type=str, default='./output',
                       help='Output directory')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use')
    parser.add_argument('--wandb', action='store_true',
                       help='Use Weights & Biases logging')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Setup dataset
    print("Setting up dataset...")
    train_dataloader, val_dataloader, tokenizer = setup_dataset(config)
    print(f"Train samples: {len(train_dataloader.dataset)}")
    print(f"Val samples: {len(val_dataloader.dataset)}")
    
    # Load model from phase 1 checkpoint
    print(f"Loading model from phase 1 checkpoint: {args.phase1_checkpoint}")
    model = BrainToTextModel.from_pretrained(args.phase1_checkpoint)
    model.to(args.device)
    
    # Apply Phase 2 configuration (as it would be after phase 2)
    model.freeze_encoder()
    model.freeze_llm()
    model.rvq.requires_grad_(True)
    
    # Disable EMA for RVQ (as done in phase 2)
    for quantizer in model.rvq.quantizers:
        quantizer.use_ema = False
        quantizer.embeddings.weight.requires_grad_(True)
    
    print(f"Total parameters: {model.get_num_params():,}")
    
    # Save as phase2_complete
    print("\nSaving model as phase2_complete...")
    save_phase2_checkpoint(model, args.output_dir)
    
    # Create multi-phase trainer
    trainer = MultiPhaseTrainer(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        device=args.device,
        output_dir=args.output_dir,
        phase1_steps=config['training']['phase1_steps'],
        phase2_steps=config['training']['phase2_steps'],
        phase3_steps=config['training']['phase3_steps'],
        num_epochs=config['training'].get('num_epochs'),
        log_wandb=args.wandb,
        warmup_ratio=config['training'].get('warmup_ratio', 0.1)
    )
    
    # Skip phases 1 and 2, start from phase 3
    print("\nSkipping Phase 1 and 2 (already completed)")
    print("Starting Phase 3: End-to-end fine-tuning")
    
    # Manually set phase
    trainer.current_phase = 2  # Will be incremented to 3 in train_phase3_full
    
    # Train phase 3
    trainer.train_phase3_full()
    
    if hasattr(trainer, 'train_phase4_distill'):
        print("\nPhase 4: Distillation (if applicable)")
        trainer.train_phase4_distill()
    
    print("\nTraining complete!")


if __name__ == "__main__":
    main()