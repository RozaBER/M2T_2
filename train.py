"""
Main training script for Brain-to-Text Model
"""
import argparse
import os
import torch
from torch.utils.data import DataLoader, random_split
from transformers import AutoTokenizer
import yaml
from pathlib import Path

from src.models.full_model import BrainToTextModel
from src.data.meg_dataset import MEGDataset, MEGDataCollator
from src.data.preprocessing import MEGPreprocessor
from src.data.tokenizer import BrainToTextTokenizer
from src.training.multi_phase_trainer_optimized import OptimizedMultiPhaseTrainer
from src.data.augmentation import MEGAugmentation, AugmentedMEGDataset
from src.training.trainer import BrainToTextTrainer
from src.training.distillation import DistillationTrainer


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def setup_dataset(config: dict):
    """Setup MEG dataset and dataloaders"""
    # Initialize preprocessor
    preprocessor = MEGPreprocessor(
        sampling_rate=config['data']['sampling_rate'],
        lowpass=config['data']['lowpass'],
        highpass=config['data']['highpass'],
        normalize=config['data']['normalize']
    )
    
    # Load tokenizer
    if config['tokenizer']['use_pretrained']:
        tokenizer = AutoTokenizer.from_pretrained(config['tokenizer']['model_name'])
        # Set padding token if not already set
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
    else:
        tokenizer = BrainToTextTokenizer.from_pretrained(config['tokenizer']['path'])
    
    # Create dataset
    dataset = MEGDataset(
        data_root=config['data']['data_root'],
        subjects=config['data'].get('subjects'),
        sessions=config['data'].get('sessions'),
        tasks=config['data'].get('tasks'),
        sampling_rate=config['data']['sampling_rate'],
        segment_length=config['data']['segment_length'],
        overlap=config['data']['overlap'],
        preprocess=True,
        max_text_length=config['model']['max_text_length'],
        cache_dir=config['data'].get('cache_dir')
    )
    
    # Split dataset
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    # Apply data augmentation to training set
    if config['training'].get('use_augmentation', True):
        augmentation = MEGAugmentation(
            time_shift_range=0.1,  # ±100ms
            channel_dropout_prob=0.1,
            gaussian_noise_std=0.01,
            apply_prob=0.5
        )
        train_dataset = AugmentedMEGDataset(train_dataset, augmentation)
    
    # Create data collator
    collator = MEGDataCollator(tokenizer, max_length=config['model']['max_text_length'])
    
    # Create dataloaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        collate_fn=collator,
        num_workers=config['data']['num_workers'],
        pin_memory=True,
        drop_last=True  # Drop last batch for stable training
    )
    
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        collate_fn=collator,
        num_workers=config['data']['num_workers'],
        pin_memory=True
    )
    
    return train_dataloader, val_dataloader, tokenizer


def setup_model(config: dict, tokenizer):
    """Initialize model"""
    # Get vocab size
    if hasattr(tokenizer, 'vocab_size'):
        vocab_size = tokenizer.vocab_size
    else:
        vocab_size = len(tokenizer)
    
    # Create model configuration
    model_config = {
        'n_channels': config['model']['n_channels'],
        'eeg_d_model': config['model']['eeg_hidden_dim'],
        'eeg_n_layers': config['model']['num_encoder_layers'],
        'num_quantizers': config['model']['n_codebooks'],
        'codebook_size': config['model']['codebook_size'],
        'vocab_size': vocab_size,
        'max_position_embeddings': config['model']['max_seq_length'],
        'freeze_llm': config['model']['freeze_llm_initially']
    }
    
    # Initialize model
    model = BrainToTextModel(**model_config)
    
    # Load checkpoint if specified
    if config['model'].get('checkpoint_path'):
        print(f"Loading checkpoint from {config['model']['checkpoint_path']}")
        model = BrainToTextModel.from_pretrained(config['model']['checkpoint_path'])
    
    return model


def main():
    parser = argparse.ArgumentParser(description='Train Brain-to-Text Model')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--mode', type=str, default='full', 
                       choices=['full', 'multi_phase', 'distill'],
                       help='Training mode')
    parser.add_argument('--teacher_checkpoint', type=str, default=None,
                       help='Teacher model checkpoint for distillation')
    parser.add_argument('--output_dir', type=str, default='./output',
                       help='Output directory')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use')
    parser.add_argument('--wandb', action='store_true',
                       help='Use Weights & Biases logging')
    parser.add_argument('--resume', type=str, default=None,
                       help='Resume from checkpoint')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Override config with command line arguments
    config['training']['output_dir'] = args.output_dir
    config['training']['device'] = args.device
    config['training']['log_wandb'] = args.wandb
    
    # Setup dataset
    print("Setting up dataset...")
    train_dataloader, val_dataloader, tokenizer = setup_dataset(config)
    print(f"Train samples: {len(train_dataloader.dataset)}")
    print(f"Val samples: {len(val_dataloader.dataset)}")
    
    # Setup model
    print("Setting up model...")
    model = setup_model(config, tokenizer)
    
    # Move model to device
    model = model.to(args.device)
    
    print(f"Total parameters: {model.get_num_params():,}")
    print(f"Trainable parameters: {model.get_num_params(only_trainable=True):,}")
    
    # Select training mode
    if args.mode == 'multi_phase':
        print("Starting optimized multi-phase training...")
        trainer = OptimizedMultiPhaseTrainer(
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
            warmup_ratio=config['training'].get('warmup_ratio', 0.1),
            learning_rate=config['training']['learning_rate'],
            weight_decay=config['training']['weight_decay'],
            gradient_accumulation_steps=config['training']['accumulation_steps'],
            fp16=config['training']['mixed_precision'],
            gradient_checkpointing=config['training'].get('gradient_checkpointing', True),
            save_steps=config['training']['save_steps'],
            eval_steps=config['training']['eval_steps'],
            logging_steps=config['training']['logging_steps']
        )
        trainer.train_all_phases()
        
    elif args.mode == 'distill':
        if not args.teacher_checkpoint:
            raise ValueError("Teacher checkpoint required for distillation")
        
        print("Loading teacher model...")
        teacher_model = BrainToTextModel.from_pretrained(args.teacher_checkpoint)
        
        print("Starting distillation training...")
        trainer = DistillationTrainer(
            student_model=model,
            teacher_model=teacher_model,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            device=args.device,
            output_dir=args.output_dir,
            temperature=config['training']['distill_temperature'],
            alpha=config['training']['distill_alpha'],
            max_steps=config['training']['max_steps'],
            num_epochs=config['training']['num_epochs'],
            gradient_clip=config['training']['gradient_clip'],
            accumulation_steps=config['training']['accumulation_steps'],
            mixed_precision=config['training']['mixed_precision'],
            log_wandb=args.wandb
        )
        trainer.train()
        
    else:  # full training
        print("Starting full model training...")
        trainer = BrainToTextTrainer(
            model=model,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            device=args.device,
            output_dir=args.output_dir,
            max_steps=config['training'].get('max_steps'),
            num_epochs=config['training'].get('num_epochs'),
            gradient_clip=config['training']['gradient_clip'],
            accumulation_steps=config['training']['accumulation_steps'],
            mixed_precision=config['training']['mixed_precision'],
            save_steps=config['training']['save_steps'],
            eval_steps=config['training']['eval_steps'],
            logging_steps=config['training']['logging_steps'],
            warmup_steps=config['training']['warmup_steps'],
            log_wandb=args.wandb
        )
        
        # Resume if specified
        if args.resume:
            print(f"Resuming from {args.resume}")
            trainer.load_checkpoint(args.resume)
        
        trainer.train()
    
    print("Training complete!")


if __name__ == "__main__":
    main()
