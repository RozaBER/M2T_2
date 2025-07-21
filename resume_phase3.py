#!/usr/bin/env python3
"""
Resume training from Phase 3 after Phase 2 completion
"""

import argparse
import torch
import os
from pathlib import Path

from train import load_config, setup_dataset
from src.models.full_model import BrainToTextModel
from src.training.multi_phase_trainer import MultiPhaseTrainer


def main():
    parser = argparse.ArgumentParser(description='Resume training from Phase 3')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to configuration file')
    parser.add_argument('--phase2_checkpoint', type=str, 
                       default='./output/phase2_complete',
                       help='Path to phase 2 checkpoint directory')
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
    
    # Load model from phase 2 checkpoint
    print(f"Loading model from phase 2 checkpoint: {args.phase2_checkpoint}")
    model = BrainToTextModel.from_pretrained(args.phase2_checkpoint)
    model.to(args.device)
    
    print(f"Total parameters: {model.get_num_params():,}")
    print(f"Trainable parameters: {model.get_num_params(only_trainable=True):,}")
    
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
    
    # Phase 4 is optional (distillation)
    print("\nPhase 3 complete! Phase 4 (distillation) requires a teacher model.")
    print("If you want to run phase 4, use --mode distill with a teacher checkpoint.")
    
    print("\nTraining complete!")


if __name__ == "__main__":
    main()