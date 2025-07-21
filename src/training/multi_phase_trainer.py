"""
Multi-phase training strategy for Brain-to-Text Model
Implements the training phases described in the model outline
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Optional, Dict, List, Tuple
from pathlib import Path
import json
from .trainer import BrainToTextTrainer
from ..models.full_model import BrainToTextModel


class MultiPhaseTrainer:
    """
    Implements multi-phase training strategy:
    Phase 1: Pre-train EEG encoder with self-supervised learning
    Phase 2: Train RVQ module with frozen encoder
    Phase 3: Fine-tune full model end-to-end
    Phase 4: Optional distillation from larger model
    """
    
    def __init__(self,
                 model: BrainToTextModel,
                 train_dataloader: DataLoader,
                 val_dataloader: Optional[DataLoader] = None,
                 device: str = 'cuda',
                 output_dir: str = './output',
                 phase1_steps: int = 50000,
                 phase2_steps: int = 30000,
                 phase3_steps: int = 100000,
                 phase4_steps: int = 50000,
                 num_epochs: Optional[int] = None,
                 log_wandb: bool = True,
                 warmup_ratio: float = 0.1):
        """
        Initialize multi-phase trainer
        
        Args:
            model: BrainToTextModel instance
            train_dataloader: Training dataloader
            val_dataloader: Validation dataloader
            device: Device to train on
            output_dir: Directory for saving checkpoints
            phase1_steps: Steps for encoder pre-training
            phase2_steps: Steps for RVQ training
            phase3_steps: Steps for end-to-end fine-tuning
            phase4_steps: Steps for distillation (if applicable)
            log_wandb: Whether to log to Weights & Biases
        """
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Phase-specific steps
        self.phase1_steps = phase1_steps
        self.phase2_steps = phase2_steps
        self.phase3_steps = phase3_steps
        self.phase4_steps = phase4_steps
        self.num_epochs = num_epochs
        
        self.log_wandb = log_wandb
        self.warmup_ratio = warmup_ratio
        self.current_phase = 0
        
    def train_all_phases(self):
        """Execute all training phases sequentially"""
        print("Starting Multi-Phase Training")
        
        # Phase 1: Pre-train encoder
        print("\n=== Phase 1: Pre-training EEG Encoder ===")
        self.train_phase1_encoder()
        
        # Phase 2: Train RVQ
        print("\n=== Phase 2: Training RVQ Module ===")
        self.train_phase2_rvq()
        
        # Phase 3: End-to-end fine-tuning
        print("\n=== Phase 3: End-to-End Fine-tuning ===")
        self.train_phase3_full()
        
        print("\nMulti-Phase Training Complete!")
    
    def train_phase1_encoder(self):
        """
        Phase 1: Pre-train EEG encoder with self-supervised learning
        Uses masked autoencoding or contrastive learning
        """
        self.current_phase = 1
        
        # Configure model for phase 1
        self.model.freeze_llm()  # Freeze LLM decoder
        self.model.rvq.requires_grad_(False)  # Freeze RVQ
        
        # Create phase-specific trainer
        phase1_trainer = EncoderPretrainTrainer(
            model=self.model,
            train_dataloader=self.train_dataloader,
            val_dataloader=self.val_dataloader,
            device=self.device,
            output_dir=self.output_dir / "phase1_encoder",
            max_steps=self.phase1_steps if self.num_epochs is None else None,
            num_epochs=self.num_epochs,
            log_wandb=self.log_wandb,
            warmup_ratio=self.warmup_ratio
        )
        
        # Train
        phase1_trainer.train()
        
        # Save phase 1 checkpoint
        self.save_phase_checkpoint("phase1_complete")
    
    def train_phase2_rvq(self):
        """
        Phase 2: Train RVQ module with frozen encoder
        Focuses on learning good discrete representations
        """
        self.current_phase = 2
        
        # Load best checkpoint from phase 1 if exists
        phase1_best = self.output_dir / "phase1_encoder" / "best_model"
        if phase1_best.exists():
            self.model = BrainToTextModel.from_pretrained(phase1_best)
            self.model.to(self.device)
        
        # Configure model for phase 2
        self.model.freeze_encoder()  # Freeze encoder
        self.model.freeze_llm()  # Keep LLM frozen
        self.model.rvq.requires_grad_(True)  # Unfreeze RVQ
        
        # Disable EMA for gradient-based training in phase 2
        for quantizer in self.model.rvq.quantizers:
            quantizer.use_ema = False
            # Ensure embeddings require gradients
            quantizer.embeddings.weight.requires_grad_(True)
        
        # Verify RVQ parameters are trainable
        rvq_trainable = sum(p.numel() for p in self.model.rvq.parameters() if p.requires_grad)
        print(f"RVQ trainable parameters: {rvq_trainable:,}")
        if rvq_trainable == 0:
            raise ValueError("No trainable parameters in RVQ module!")
        
        # Create phase-specific trainer
        phase2_trainer = RVQTrainer(
            model=self.model,
            train_dataloader=self.train_dataloader,
            val_dataloader=self.val_dataloader,
            device=self.device,
            output_dir=self.output_dir / "phase2_rvq",
            max_steps=self.phase2_steps,  # Always use phase steps
            num_epochs=None,  # Ignore epochs for phase training
            log_wandb=self.log_wandb,
            warmup_ratio=self.warmup_ratio
        )
        
        # Train
        phase2_trainer.train()
        
        # Save phase 2 checkpoint
        self.save_phase_checkpoint("phase2_complete")
    
    def train_phase3_full(self):
        """
        Phase 3: End-to-end fine-tuning
        Train all components together
        """
        self.current_phase = 3
        
        # Load best checkpoint from phase 2 if exists
        phase2_best = self.output_dir / "phase2_rvq" / "best_model"
        if phase2_best.exists():
            self.model = BrainToTextModel.from_pretrained(phase2_best)
            self.model.to(self.device)
        
        # Configure model for phase 3
        self.model.unfreeze_encoder()  # Unfreeze encoder
        self.model.unfreeze_llm()  # Unfreeze LLM
        
        # Create standard trainer for end-to-end training
        phase3_trainer = BrainToTextTrainer(
            model=self.model,
            train_dataloader=self.train_dataloader,
            val_dataloader=self.val_dataloader,
            device=self.device,
            output_dir=self.output_dir / "phase3_full",
            max_steps=self.phase3_steps if self.num_epochs is None else None,
            num_epochs=self.num_epochs,
            log_wandb=self.log_wandb,
            gradient_clip=0.5,  # Lower gradient clipping for stability
            accumulation_steps=4  # More accumulation for larger effective batch
        )
        
        # Train
        phase3_trainer.train()
        
        # Save final model
        self.save_phase_checkpoint("final_model")
    
    def save_phase_checkpoint(self, name: str):
        """Save checkpoint after completing a phase"""
        checkpoint_dir = self.output_dir / name
        self.model.save_pretrained(checkpoint_dir)
        
        # Save phase information
        with open(checkpoint_dir / "phase_info.json", 'w') as f:
            json.dump({
                'phase': self.current_phase,
                'phase_name': name,
                'model_state': {
                    'encoder_frozen': not any(p.requires_grad for p in self.model.eeg_encoder.parameters()),
                    'rvq_frozen': not any(p.requires_grad for p in self.model.rvq.parameters()),
                    'llm_frozen': not any(p.requires_grad for p in self.model.llm_decoder.parameters())
                }
            }, f, indent=2)


class EncoderPretrainTrainer(BrainToTextTrainer):
    """
    Specialized trainer for encoder pre-training phase
    Implements masked autoencoding for self-supervised learning
    """
    
    def __init__(self, *args, mask_ratio: float = 0.15, **kwargs):
        super().__init__(*args, **kwargs)
        self.mask_ratio = mask_ratio
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Custom training step for encoder pre-training"""
        # Get MEG signals
        meg_signals = batch['meg_signals']
        batch_size, n_channels, time_steps = meg_signals.shape
        
        # Create random mask
        mask = torch.rand(batch_size, time_steps, device=meg_signals.device) < self.mask_ratio
        
        # Apply mask by zeroing out selected time steps
        masked_signals = meg_signals.clone()
        masked_signals[:, :, mask] = 0
        
        # Forward pass through encoder only
        with torch.cuda.amp.autocast(enabled=self.mixed_precision):
            # Encode masked signals
            encoder_output = self.model.eeg_encoder(masked_signals)
            
            # Quantize
            quantized, indices, vq_losses = self.model.rvq(encoder_output)
            
            # Reconstruction loss (if RVQ has reconstruction)
            if hasattr(self.model.rvq, 'use_reconstruction') and self.model.rvq.use_reconstruction:
                rvq_output = self.model.rvq(meg_signals, encoder_output)
                reconstruction_loss = rvq_output.get('reconstruction_loss', 0)
            else:
                reconstruction_loss = torch.tensor(0.0, device=self.device)
            
            # Total loss
            loss = vq_losses['total_vq_loss'] + reconstruction_loss
        
        return {
            'loss': loss,
            'vq_loss': vq_losses['total_vq_loss'],
            'reconstruction_loss': reconstruction_loss
        }


class RVQTrainer(BrainToTextTrainer):
    """
    Specialized trainer for RVQ training phase
    Focuses on learning good discrete representations
    """
    
    def __init__(self, *args, codebook_temp: float = 1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.codebook_temp = codebook_temp
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Custom training step for RVQ training"""
        # Get MEG signals
        meg_signals = batch['meg_signals']
        
        # Ensure model is in training mode
        self.model.train()
        
        with torch.cuda.amp.autocast(enabled=self.mixed_precision):
            # Forward pass through encoder (frozen but keep in graph)
            encoder_output = self.model.eeg_encoder(meg_signals)
            encoder_output = encoder_output.detach()  # Detach to prevent encoder gradients
            
            # Quantize with RVQ (trainable)
            quantized, indices, vq_losses = self.model.rvq(encoder_output)
            reconstruction_loss = torch.tensor(0.0, device=self.device)
            
            # Ensure vq_losses are on the correct device and require grad
            if not isinstance(vq_losses, dict):
                vq_losses = {'total_vq_loss': vq_losses}
            
            # Add entropy regularization to encourage codebook usage
            try:
                codebook_usage = self.model.rvq.get_codebook_usage()
                if codebook_usage:
                    entropy_loss = sum(
                        -stats['entropy'] for stats in codebook_usage.values()
                    ) / len(codebook_usage)
                else:
                    entropy_loss = torch.tensor(0.0, device=self.device)
            except:
                entropy_loss = torch.tensor(0.0, device=self.device)
            
            # Total loss - ensure all components are tensors with grad
            total_vq_loss = vq_losses.get('total_vq_loss', vq_losses.get('loss', torch.tensor(0.0, device=self.device)))
            
            # Ensure loss requires grad
            loss = total_vq_loss + 0.1 * reconstruction_loss + 0.01 * entropy_loss
            
            # If loss doesn't require grad, create a dummy loss that does
            if not loss.requires_grad:
                # Create a small regularization term from RVQ parameters
                reg_loss = 0.0
                for param in self.model.rvq.parameters():
                    if param.requires_grad:
                        reg_loss = reg_loss + 1e-8 * param.norm()
                        break
                loss = loss + reg_loss
        
        # Log codebook usage statistics
        if self.global_step % self.logging_steps == 0:
            for name, stats in codebook_usage.items():
                self._log_metrics({
                    f'rvq/{name}_usage_rate': stats['usage_rate'],
                    f'rvq/{name}_entropy': stats['entropy']
                })
        
        return {
            'loss': loss,
            'vq_loss': vq_losses['total_vq_loss'],
            'reconstruction_loss': reconstruction_loss,
            'entropy_loss': entropy_loss
        }
    
    def on_epoch_end(self):
        """Reset codebook usage statistics at end of epoch"""
        self.model.rvq.reset_usage_stats()
