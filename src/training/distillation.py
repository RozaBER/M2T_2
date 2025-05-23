"""
Knowledge Distillation trainer for Brain-to-Text Model
Distills knowledge from a larger teacher model to the student model
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Optional, Dict, List, Tuple
from pathlib import Path
from .trainer import BrainToTextTrainer
from ..models.full_model import BrainToTextModel


class DistillationTrainer(BrainToTextTrainer):
    """
    Implements knowledge distillation from a teacher model
    Can be used in Phase 4 of multi-phase training
    """
    
    def __init__(self,
                 student_model: BrainToTextModel,
                 teacher_model: BrainToTextModel,
                 train_dataloader: DataLoader,
                 val_dataloader: Optional[DataLoader] = None,
                 temperature: float = 3.0,
                 alpha: float = 0.7,
                 distill_encoder: bool = True,
                 distill_rvq: bool = True,
                 distill_llm: bool = True,
                 device: str = 'cuda',
                 **kwargs):
        """
        Initialize distillation trainer
        
        Args:
            student_model: Student BrainToTextModel to train
            teacher_model: Teacher BrainToTextModel (pre-trained)
            train_dataloader: Training dataloader
            val_dataloader: Validation dataloader
            temperature: Distillation temperature
            alpha: Weight for distillation loss (vs task loss)
            distill_encoder: Whether to distill encoder representations
            distill_rvq: Whether to distill RVQ representations
            distill_llm: Whether to distill LLM outputs
            device: Device to train on
            **kwargs: Additional arguments for BrainToTextTrainer
        """
        # Initialize base trainer with student model
        super().__init__(
            model=student_model,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            device=device,
            **kwargs
        )
        
        self.teacher_model = teacher_model.to(device)
        self.teacher_model.eval()  # Teacher always in eval mode
        
        # Distillation parameters
        self.temperature = temperature
        self.alpha = alpha
        self.distill_encoder = distill_encoder
        self.distill_rvq = distill_rvq
        self.distill_llm = distill_llm
        
        # Freeze teacher model
        for param in self.teacher_model.parameters():
            param.requires_grad = False
    
    def compute_distillation_loss(self,
                                 student_outputs: Dict[str, torch.Tensor],
                                 teacher_outputs: Dict[str, torch.Tensor],
                                 batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Compute various distillation losses
        
        Args:
            student_outputs: Outputs from student model
            teacher_outputs: Outputs from teacher model
            batch: Input batch
            
        Returns:
            Dictionary of distillation losses
        """
        losses = {}
        
        # Encoder distillation
        if self.distill_encoder and 'encoder_hidden' in student_outputs:
            encoder_loss = F.mse_loss(
                student_outputs['encoder_hidden'],
                teacher_outputs['encoder_hidden'].detach()
            )
            losses['encoder_distill_loss'] = encoder_loss
        
        # RVQ distillation (codebook alignment)
        if self.distill_rvq and 'eeg_indices' in student_outputs:
            # Soft targets from teacher codebook distances
            with torch.no_grad():
                teacher_quantized = teacher_outputs.get('quantized_eeg')
                if teacher_quantized is not None:
                    rvq_loss = F.mse_loss(
                        student_outputs['quantized_eeg'],
                        teacher_quantized
                    )
                    losses['rvq_distill_loss'] = rvq_loss
        
        # LLM output distillation
        if self.distill_llm and 'logits' in student_outputs:
            student_logits = student_outputs['logits']
            teacher_logits = teacher_outputs['logits'].detach()
            
            # KL divergence with temperature
            student_log_probs = F.log_softmax(student_logits / self.temperature, dim=-1)
            teacher_probs = F.softmax(teacher_logits / self.temperature, dim=-1)
            
            kl_loss = F.kl_div(
                student_log_probs,
                teacher_probs,
                reduction='batchmean'
            ) * (self.temperature ** 2)
            
            losses['llm_distill_loss'] = kl_loss
        
        return losses
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Custom training step with distillation"""
        # Move batch to device
        batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                for k, v in batch.items()}
        
        # Get teacher outputs (no gradient)
        with torch.no_grad():
            teacher_outputs = self._get_model_outputs(self.teacher_model, batch)
        
        # Get student outputs (with gradient)
        with torch.cuda.amp.autocast(enabled=self.mixed_precision):
            student_outputs = self._get_model_outputs(self.model, batch)
            
            # Task loss
            task_loss = student_outputs['loss']
            
            # Distillation losses
            distill_losses = self.compute_distillation_loss(
                student_outputs, teacher_outputs, batch
            )
            
            # Combine losses
            total_distill_loss = sum(distill_losses.values())
            
            # Total loss with weighting
            loss = (1 - self.alpha) * task_loss + self.alpha * total_distill_loss
        
        # Prepare output
        output = {
            'loss': loss / self.accumulation_steps,
            'task_loss': task_loss.item(),
            'distill_loss': total_distill_loss.item()
        }
        output.update({k: v.item() for k, v in distill_losses.items()})
        
        return output
    
    def _get_model_outputs(self, model: BrainToTextModel, 
                          batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Get outputs from model with additional intermediate representations"""
        # Standard forward pass
        outputs = model(**batch)
        
        # Get intermediate representations if needed
        if self.distill_encoder or self.distill_rvq:
            meg_signals = batch.get('meg_signals')
            if meg_signals is not None:
                # Get encoder output
                encoder_output = model.eeg_encoder(meg_signals)
                outputs['encoder_hidden'] = encoder_output
                
                # Get RVQ output
                if hasattr(model, 'encode_eeg'):
                    quantized, indices, _ = model.encode_eeg(meg_signals)
                    outputs['quantized_eeg'] = quantized
                    outputs['eeg_indices'] = indices
        
        return outputs
    
    def train(self):
        """Override train method to handle distillation-specific logging"""
        self.model.train()
        self.teacher_model.eval()  # Ensure teacher stays in eval mode
        
        # Determine total steps
        if self.max_steps:
            total_steps = self.max_steps
        elif self.num_epochs:
            total_steps = len(self.train_dataloader) * self.num_epochs
        else:
            raise ValueError("Either max_steps or num_epochs must be specified")
        
        # Training loop
        epoch_iterator = range(self.num_epochs) if self.num_epochs else range(1000)
        
        for epoch in epoch_iterator:
            self.current_epoch = epoch
            epoch_loss = 0.0
            epoch_task_loss = 0.0
            epoch_distill_loss = 0.0
            
            for step, batch in enumerate(self.train_dataloader):
                # Training step
                step_outputs = self.train_step(batch)
                loss = step_outputs['loss']
                
                # Backward pass
                if self.mixed_precision:
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()
                
                # Gradient accumulation
                if (step + 1) % self.accumulation_steps == 0:
                    # Gradient clipping
                    if self.gradient_clip > 0:
                        if self.mixed_precision:
                            self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), self.gradient_clip
                        )
                    
                    # Optimizer step
                    if self.mixed_precision:
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        self.optimizer.step()
                    
                    self.scheduler.step()
                    self.optimizer.zero_grad()
                    
                    self.global_step += 1
                
                # Accumulate losses
                epoch_loss += loss.item() * self.accumulation_steps
                epoch_task_loss += step_outputs['task_loss']
                epoch_distill_loss += step_outputs['distill_loss']
                
                # Logging
                if self.global_step % self.logging_steps == 0:
                    avg_loss = epoch_loss / (step + 1)
                    avg_task_loss = epoch_task_loss / (step + 1)
                    avg_distill_loss = epoch_distill_loss / (step + 1)
                    
                    log_dict = {
                        'train/loss': avg_loss,
                        'train/task_loss': avg_task_loss,
                        'train/distill_loss': avg_distill_loss,
                        'train/learning_rate': self.scheduler.get_last_lr()[0]
                    }
                    
                    # Add component-specific losses
                    for key in ['encoder_distill_loss', 'rvq_distill_loss', 'llm_distill_loss']:
                        if key in step_outputs:
                            log_dict[f'train/{key}'] = step_outputs[key]
                    
                    self._log_metrics(log_dict)
                
                # Evaluation
                if self.val_dataloader and self.global_step % self.eval_steps == 0:
                    val_metrics = self.evaluate()
                    self._log_metrics({f'val/{k}': v for k, v in val_metrics.items()})
                    
                    # Save best model
                    if self._is_better_metric(val_metrics):
                        self.save_checkpoint('best_model')
                
                # Save checkpoint
                if self.global_step % self.save_steps == 0:
                    self.save_checkpoint(f'checkpoint-{self.global_step}')
                
                # Check if done
                if self.max_steps and self.global_step >= self.max_steps:
                    return
            
            # End of epoch
            print(f"Epoch {epoch} - Loss: {epoch_loss/(step+1):.4f}, "
                  f"Task: {epoch_task_loss/(step+1):.4f}, "
                  f"Distill: {epoch_distill_loss/(step+1):.4f}")


class ProgressiveDistillationTrainer(DistillationTrainer):
    """
    Progressive distillation that gradually shifts from teacher to student
    """
    
    def __init__(self, *args, 
                 initial_alpha: float = 0.9,
                 final_alpha: float = 0.1,
                 warmup_steps: int = 1000,
                 **kwargs):
        super().__init__(*args, alpha=initial_alpha, **kwargs)
        self.initial_alpha = initial_alpha
        self.final_alpha = final_alpha
        self.warmup_steps = warmup_steps
    
    def update_alpha(self):
        """Update distillation weight based on training progress"""
        if self.global_step < self.warmup_steps:
            # Linear warmup
            progress = self.global_step / self.warmup_steps
            self.alpha = self.initial_alpha * (1 - progress) + self.final_alpha * progress
        else:
            self.alpha = self.final_alpha
