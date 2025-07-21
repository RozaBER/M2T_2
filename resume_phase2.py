#!/usr/bin/env python3
"""
Resume training from Phase 2 without repeating Phase 1
"""

import argparse
import torch
import os
from pathlib import Path

from train import load_config, setup_dataset
from src.models.full_model import BrainToTextModel
from src.training.multi_phase_trainer import MultiPhaseTrainer


def main():
    parser = argparse.ArgumentParser(description='Resume training from Phase 2')
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
    
    # Skip phase 1 and start from phase 2
    print("\nSkipping Phase 1 (already completed)")
    print("Starting from Phase 2: Training RVQ Module")
    
    # Manually set phase and continue
    trainer.current_phase = 1  # Will be incremented to 2 in train_phase2_rvq
    
    # Train phase 2
    trainer.train_phase2_rvq()
    
    # Continue with remaining phases
    print("\nContinuing with Phase 3: End-to-end fine-tuning")
    trainer.train_phase3_full()
    
    if hasattr(trainer, 'train_phase4_distill'):
        print("\nPhase 4: Distillation (if applicable)")
        trainer.train_phase4_distill()
    
    print("\nTraining complete!")


if __name__ == "__main__":
    main()