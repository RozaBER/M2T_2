"""
Optimized Multi-phase training strategy for Brain-to-Text Model
Implements the training phases with proper loss balancing and monitoring
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Optional, Dict, List, Tuple
from pathlib import Path
import json
import wandb
from tqdm import tqdm
import numpy as np
from .trainer import BrainToTextTrainer
from ..models.full_model import BrainToTextModel
from .monitoring import TrainingMonitor, EarlyStopping, RVQMonitor


class OptimizedMultiPhaseTrainer:
    """
    Implements optimized multi-phase training strategy:
    Phase 1: Pre-train encoder + RVQ with reconstruction
    Phase 2: Train RVQ for diversity with frozen encoder
    Phase 3: Fine-tune full model end-to-end
    Phase 4: Optional distillation from larger model
    """
    
    def __init__(self,
                 model: BrainToTextModel,
                 train_dataloader: DataLoader,
                 val_dataloader: Optional[DataLoader] = None,
                 device: str = 'cuda',
                 output_dir: str = './output',
                 phase1_steps: int = 10000,
                 phase2_steps: int = 5000,
                 phase3_steps: int = 20000,
                 phase4_steps: int = 10000,
                 num_epochs: Optional[int] = None,
                 log_wandb: bool = True,
                 warmup_ratio: float = 0.1,
                 learning_rate: float = 1e-4,
                 weight_decay: float = 0.01,
                 gradient_accumulation_steps: int = 4,
                 fp16: bool = True,
                 gradient_checkpointing: bool = True,
                 save_steps: int = 1000,
                 eval_steps: int = 500,
                 logging_steps: int = 10):
        """
        Initialize optimized multi-phase trainer
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
        
        # Training parameters
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.fp16 = fp16
        self.gradient_checkpointing = gradient_checkpointing
        self.save_steps = save_steps
        self.eval_steps = eval_steps
        self.logging_steps = logging_steps
        
        self.log_wandb = log_wandb
        self.warmup_ratio = warmup_ratio
        self.current_phase = 0
        
        # Enable gradient checkpointing if requested
        if gradient_checkpointing:
            # Check if encoder supports gradient checkpointing
            if hasattr(self.model.eeg_encoder, 'gradient_checkpointing_enable'):
                self.model.eeg_encoder.gradient_checkpointing_enable()
            elif hasattr(self.model.eeg_encoder, 'gradient_checkpointing'):
                self.model.eeg_encoder.gradient_checkpointing = True
            else:
                print("Warning: EEG encoder doesn't support gradient checkpointing")
            
            # Check if LLM decoder supports gradient checkpointing
            if hasattr(self.model.llm_decoder, 'gradient_checkpointing_enable'):
                try:
                    self.model.llm_decoder.gradient_checkpointing_enable()
                except ValueError as e:
                    # Some models need the gradient_checkpointing attribute set first
                    if hasattr(self.model.llm_decoder, 'gradient_checkpointing'):
                        self.model.llm_decoder.gradient_checkpointing = True
                    else:
                        print(f"Warning: LLM decoder doesn't support gradient checkpointing: {e}")
            elif hasattr(self.model.llm_decoder, 'gradient_checkpointing'):
                self.model.llm_decoder.gradient_checkpointing = True
        
        # Initialize scaler for mixed precision
        self.scaler = torch.cuda.amp.GradScaler() if fp16 else None
        
        # Initialize wandb
        if self.log_wandb:
            try:
                import os
                # Set wandb to offline mode if no API key is available
                if not os.getenv("WANDB_API_KEY") and not os.path.exists(os.path.expanduser("~/.netrc")):
                    os.environ["WANDB_MODE"] = "offline"
                    print("wandb: Using offline mode (no API key found)")
                
                wandb.init(
                    project="brain-to-text-optimized",
                    name=f"run_{self.output_dir.name}",
                    config={
                        "phase1_steps": phase1_steps,
                        "phase2_steps": phase2_steps,
                        "phase3_steps": phase3_steps,
                        "learning_rate": learning_rate,
                        "batch_size": train_dataloader.batch_size,
                        "gradient_accumulation_steps": gradient_accumulation_steps,
                        "fp16": fp16,
                        "gradient_checkpointing": gradient_checkpointing,
                    },
                    mode=os.getenv("WANDB_MODE", "online")
                )
            except Exception as e:
                print(f"Warning: Failed to initialize wandb: {e}")
                print("Continuing without wandb logging...")
                self.log_wandb = False
        
        # Initialize monitors
        self.training_monitor = TrainingMonitor(
            output_dir=str(self.output_dir),
            log_wandb=log_wandb
        )
        
        # Get RVQ configuration
        rvq_config = self.model.rvq.rvq if hasattr(self.model.rvq, 'rvq') else self.model.rvq
        self.rvq_monitor = RVQMonitor(
            num_quantizers=rvq_config.num_quantizers,
            codebook_size=rvq_config.codebook_size
        )
        
        # Initialize early stopping
        self.early_stopping = EarlyStopping(
            patience=20,
            min_delta=0.001,
            mode='min',
            restore_best_weights=True
        )
    
    def train_all_phases(self):
        """Execute all training phases sequentially"""
        print("Starting Optimized Multi-Phase Training")
        
        # Phase 1: Pre-train encoder + RVQ with reconstruction
        print("\n=== Phase 1: Pre-training Encoder + RVQ with Reconstruction ===")
        self.train_phase1_encoder_rvq()
        
        # Phase 2: Train RVQ for diversity
        print("\n=== Phase 2: Training RVQ for Diversity ===")
        self.train_phase2_rvq_diversity()
        
        # Phase 3: End-to-end fine-tuning
        print("\n=== Phase 3: End-to-End Fine-tuning ===")
        self.train_phase3_full()
        
        print("\nOptimized Multi-Phase Training Complete!")
    
    def train_phase1_encoder_rvq(self):
        """
        Phase 1: Pre-train encoder + RVQ with reconstruction loss
        Focus on learning good representations
        """
        self.current_phase = 1
        
        # Configure model for phase 1
        self.model.freeze_llm()  # Freeze LLM decoder
        self.model.eeg_encoder.requires_grad_(True)
        self.model.rvq.requires_grad_(True)
        
        # Enable reconstruction in RVQ if available
        if hasattr(self.model.rvq, 'use_reconstruction'):
            self.model.rvq.use_reconstruction = True
        
        # Create optimizer for encoder + RVQ
        optimizer = torch.optim.AdamW([
            {'params': self.model.eeg_encoder.parameters(), 'lr': self.learning_rate},
            {'params': self.model.rvq.parameters(), 'lr': self.learning_rate * 0.5}
        ], weight_decay=self.weight_decay)
        
        # Learning rate scheduler
        num_training_steps = self.phase1_steps
        num_warmup_steps = int(num_training_steps * self.warmup_ratio)
        scheduler = self._get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps, num_training_steps
        )
        
        # Training loop
        self._train_phase(
            optimizer=optimizer,
            scheduler=scheduler,
            num_steps=self.phase1_steps,
            phase_name="Phase 1: Encoder + RVQ",
            loss_weights={
                'reconstruction': 1.0,
                'commitment': 0.25,
                'diversity': 0.1,
                'perplexity': 0.05
            }
        )
    
    def train_phase2_rvq_diversity(self):
        """
        Phase 2: Train RVQ for better codebook utilization
        Freeze encoder, focus on diversity
        """
        self.current_phase = 2
        
        # Configure model for phase 2
        self.model.eeg_encoder.requires_grad_(False)  # Freeze encoder
        self.model.rvq.requires_grad_(True)
        self.model.freeze_llm()  # Keep LLM frozen
        
        # Reset RVQ usage statistics
        self.model.rvq.reset_usage_stats()
        
        # Create optimizer for RVQ only
        optimizer = torch.optim.AdamW(
            self.model.rvq.parameters(),
            lr=self.learning_rate * 0.1,  # Lower LR for fine-tuning
            weight_decay=self.weight_decay
        )
        
        # Learning rate scheduler
        num_training_steps = self.phase2_steps
        num_warmup_steps = int(num_training_steps * 0.05)  # Short warmup
        scheduler = self._get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps, num_training_steps
        )
        
        # Training loop with emphasis on diversity
        self._train_phase(
            optimizer=optimizer,
            scheduler=scheduler,
            num_steps=self.phase2_steps,
            phase_name="Phase 2: RVQ Diversity",
            loss_weights={
                'reconstruction': 0.5,
                'commitment': 0.1,
                'diversity': 0.5,  # High weight on diversity
                'perplexity': 0.2   # High weight on perplexity
            }
        )
    
    def train_phase3_full(self):
        """
        Phase 3: End-to-end fine-tuning
        Train full model with all components
        """
        self.current_phase = 3
        
        # Configure model for phase 3
        self.model.unfreeze_all()  # Unfreeze everything
        
        # Create optimizer with different learning rates
        optimizer = torch.optim.AdamW([
            {'params': self.model.eeg_encoder.parameters(), 'lr': self.learning_rate * 0.5},
            {'params': self.model.rvq.parameters(), 'lr': self.learning_rate * 0.1},
            {'params': self.model.llm_decoder.parameters(), 'lr': self.learning_rate}
        ], weight_decay=self.weight_decay)
        
        # Learning rate scheduler
        num_training_steps = self.phase3_steps
        num_warmup_steps = int(num_training_steps * self.warmup_ratio)
        scheduler = self._get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps, num_training_steps
        )
        
        # Training loop
        self._train_phase(
            optimizer=optimizer,
            scheduler=scheduler,
            num_steps=self.phase3_steps,
            phase_name="Phase 3: End-to-End",
            loss_weights={
                'generation': 1.0,
                'reconstruction': 0.5,
                'commitment': 0.2,
                'diversity': 0.1,
                'perplexity': 0.05
            }
        )
    
    def _train_phase(self, optimizer, scheduler, num_steps, phase_name, loss_weights):
        """Generic training loop for a phase"""
        self.model.train()
        
        global_step = 0
        accumulation_counter = 0
        accumulated_loss = 0.0
        loss_components = {}
        
        # Initialize metrics for display
        avg_utilization = 0.0
        avg_perplexity = 0.0
        
        # Progress bar
        pbar = tqdm(total=num_steps, desc=phase_name)
        
        while global_step < num_steps:
            for batch in self.train_dataloader:
                if global_step >= num_steps:
                    break
                
                # Move batch to device
                batch = {k: v.to(self.device) if torch.is_tensor(v) else v 
                        for k, v in batch.items()}
                
                # Forward pass with mixed precision
                with torch.cuda.amp.autocast(enabled=self.fp16):
                    outputs = self.model(**batch)
                    
                    # Calculate weighted loss based on phase
                    loss = torch.tensor(0.0, device=self.device, dtype=torch.float32)
                    if 'loss' in outputs and self.current_phase == 3:
                        # Use model's combined loss for phase 3
                        loss = outputs['loss']
                    else:
                        # Custom loss weighting for phases 1 and 2
                        if 'reconstruction_loss' in outputs and 'reconstruction' in loss_weights:
                            weight = torch.tensor(loss_weights['reconstruction'], device=self.device, dtype=torch.float32)
                            loss += weight * outputs['reconstruction_loss']
                        if 'commitment_loss' in outputs and 'commitment' in loss_weights:
                            weight = torch.tensor(loss_weights['commitment'], device=self.device, dtype=torch.float32)
                            loss += weight * outputs['commitment_loss']
                        if 'diversity_loss' in outputs and 'diversity' in loss_weights:
                            weight = torch.tensor(loss_weights['diversity'], device=self.device, dtype=torch.float32)
                            loss += weight * outputs['diversity_loss']
                        if 'perplexity_loss' in outputs and 'perplexity' in loss_weights:
                            weight = torch.tensor(loss_weights['perplexity'], device=self.device, dtype=torch.float32)
                            loss += weight * outputs['perplexity_loss']
                        if 'lm_loss' in outputs and 'generation' in loss_weights:
                            weight = torch.tensor(loss_weights['generation'], device=self.device, dtype=torch.float32)
                            loss += weight * outputs['lm_loss']
                    
                    # Scale loss for gradient accumulation
                    loss = loss / self.gradient_accumulation_steps
                
                # Backward pass
                if self.fp16:
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()
                
                accumulated_loss += loss.item()
                
                # Track loss components
                for key in ['reconstruction_loss', 'commitment_loss', 'diversity_loss', 
                           'perplexity_loss', 'lm_loss']:
                    if key in outputs:
                        if key not in loss_components:
                            loss_components[key] = 0.0
                        loss_components[key] += outputs[key].item() / self.gradient_accumulation_steps
                
                accumulation_counter += 1
                
                # Gradient accumulation
                if accumulation_counter % self.gradient_accumulation_steps == 0:
                    # Gradient clipping
                    if self.fp16:
                        self.scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    
                    # Optimizer step
                    if self.fp16:
                        self.scaler.step(optimizer)
                        self.scaler.update()
                    else:
                        optimizer.step()
                    
                    scheduler.step()
                    optimizer.zero_grad()
                    
                    global_step += 1
                    
                    # Logging
                    if global_step % self.logging_steps == 0:
                        # Get RVQ statistics
                        rvq_stats = self.model.rvq.get_codebook_usage()
                        avg_utilization = np.mean([
                            stats['usage_rate'] for stats in rvq_stats.values() 
                            if isinstance(stats, dict) and 'usage_rate' in stats
                        ])
                        
                        # Get auxiliary losses from outputs
                        aux_losses = outputs.get('aux_losses', {})
                        avg_perplexity = aux_losses.get('avg_perplexity', 0.0)
                        
                        log_dict = {
                            f"{phase_name}/loss": accumulated_loss,
                            f"{phase_name}/lr": scheduler.get_last_lr()[0],
                            f"{phase_name}/rvq_utilization": avg_utilization,
                            f"{phase_name}/temperature": rvq_stats.get('overall', {}).get('temperature', 1.0),
                            f"{phase_name}/avg_perplexity": avg_perplexity
                        }
                        
                        # Add loss components
                        for key, value in loss_components.items():
                            log_dict[f"{phase_name}/{key}"] = value
                        
                        # Update monitors
                        metrics = {
                            'loss': accumulated_loss,
                            'rvq_utilization': avg_utilization,
                            'avg_perplexity': avg_perplexity,
                            'temperature': rvq_stats.get('overall', {}).get('temperature', 1.0)
                        }
                        metrics.update(loss_components)
                        
                        self.training_monitor.update(metrics, step=global_step)
                        self.rvq_monitor.update(rvq_stats, step=global_step)
                        
                        # Check RVQ health
                        rvq_health = self.rvq_monitor.check_codebook_health()
                        if not rvq_health['healthy']:
                            print(f"\nWarning: RVQ issues detected at step {global_step}:")
                            for issue in rvq_health['issues']:
                                print(f"  - {issue}")
                        
                        # Reset accumulators
                        accumulated_loss = 0.0
                        loss_components = {}
                    
                    # Evaluation
                    if global_step % self.eval_steps == 0 and self.val_dataloader:
                        should_stop = self._evaluate(phase_name, global_step)
                        if should_stop:
                            pbar.close()
                            return  # Exit training early
                    
                    # Saving
                    if global_step % self.save_steps == 0:
                        self._save_checkpoint(phase_name, global_step)
                        # Save monitor checkpoint
                        self.training_monitor.save_checkpoint(
                            self.model, optimizer, scheduler, 
                            epoch=global_step // len(self.train_dataloader), 
                            is_best=(global_step == self.early_stopping.best_epoch)
                        )
                    
                    # Update progress bar
                    pbar.update(1)
                    pbar.set_postfix({
                        'loss': f"{accumulated_loss:.4f}",
                        'lr': f"{scheduler.get_last_lr()[0]:.2e}",
                        'rvq_util': f"{avg_utilization:.2%}"
                    })
        
        pbar.close()
    
    def _evaluate(self, phase_name, global_step):
        """Evaluate model on validation set"""
        self.model.eval()
        total_loss = 0.0
        total_batches = 0
        loss_components = {}
        
        with torch.no_grad():
            for batch in self.val_dataloader:
                batch = {k: v.to(self.device) if torch.is_tensor(v) else v 
                        for k, v in batch.items()}
                
                with torch.cuda.amp.autocast(enabled=self.fp16):
                    outputs = self.model(**batch)
                
                if 'loss' in outputs:
                    total_loss += outputs['loss'].item()
                    
                # Track loss components
                for key in ['reconstruction_loss', 'commitment_loss', 'diversity_loss', 
                           'perplexity_loss', 'lm_loss']:
                    if key in outputs:
                        if key not in loss_components:
                            loss_components[key] = 0.0
                        loss_components[key] += outputs[key].item()
                
                total_batches += 1
        
        avg_loss = total_loss / total_batches if total_batches > 0 else 0.0
        
        # Average loss components
        for key in loss_components:
            loss_components[key] /= total_batches
        
        # Log evaluation metrics
        eval_metrics = {f"{phase_name}/eval_loss": avg_loss}
        for key, value in loss_components.items():
            eval_metrics[f"{phase_name}/eval_{key}"] = value
        
        if self.log_wandb:
            wandb.log(eval_metrics, step=global_step)
        
        print(f"\n{phase_name} - Eval Loss: {avg_loss:.4f}")
        
        # Check early stopping
        should_stop = self.early_stopping(avg_loss, self.model, epoch=global_step)
        if should_stop:
            print(f"\nEarly stopping triggered at step {global_step}")
            print(f"Best loss: {self.early_stopping.best_score:.4f} at step {self.early_stopping.best_epoch}")
            # Save final report
            report = self.training_monitor.generate_report()
            with open(self.output_dir / f"{phase_name}_final_report.txt", 'w') as f:
                f.write(report)
            return True  # Signal to stop training
        
        self.model.train()
        return False
    
    def _save_checkpoint(self, phase_name, global_step):
        """Save model checkpoint"""
        checkpoint_dir = self.output_dir / f"{phase_name.replace(' ', '_').replace(':', '')}_step_{global_step}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model
        self.model.save_pretrained(checkpoint_dir)
        
        # Save training state
        state = {
            'phase': self.current_phase,
            'global_step': global_step,
            'phase_name': phase_name
        }
        
        with open(checkpoint_dir / 'training_state.json', 'w') as f:
            json.dump(state, f)
        
        print(f"\nSaved checkpoint to {checkpoint_dir}")
    
    def _get_linear_schedule_with_warmup(self, optimizer, num_warmup_steps, num_training_steps):
        """Create linear schedule with warmup"""
        from transformers import get_linear_schedule_with_warmup
        return get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps
        )