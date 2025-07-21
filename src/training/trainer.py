"""
Main trainer for Brain-to-Text Model
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR
from typing import Optional, Dict, List, Tuple, Callable
import numpy as np
from tqdm import tqdm
import wandb
from pathlib import Path
import json
from datetime import datetime
from src.evaluation.metrics import EncoderEvaluationMetrics
from src.evaluation.encoder_evaluator import EncoderEvaluator


class BrainToTextTrainer:
    """
    Trainer for end-to-end Brain-to-Text model
    """
    
    def __init__(self,
                 model: nn.Module,
                 train_dataloader: DataLoader,
                 val_dataloader: Optional[DataLoader] = None,
                 optimizer: Optional[torch.optim.Optimizer] = None,
                 scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
                 device: str = 'cuda',
                 output_dir: str = './output',
                 log_wandb: bool = True,
                 gradient_clip: float = 1.0,
                 accumulation_steps: int = 1,
                 mixed_precision: bool = True,
                 save_steps: int = 1000,
                 eval_steps: int = 500,
                 logging_steps: int = 100,
                 max_steps: Optional[int] = None,
                 num_epochs: Optional[int] = None,
                 warmup_steps: int = 1000,
                 warmup_ratio: Optional[float] = None,
                 eval_metric: str = 'loss',
                 compute_generation_metrics: bool = False,
                 tokenizer = None):
        """
        Initialize trainer
        
        Args:
            model: BrainToTextModel instance
            train_dataloader: Training dataloader
            val_dataloader: Validation dataloader
            optimizer: Optimizer (created if None)
            scheduler: Learning rate scheduler
            device: Device to train on
            output_dir: Directory for saving checkpoints
            log_wandb: Whether to log to Weights & Biases
            gradient_clip: Gradient clipping value
            accumulation_steps: Gradient accumulation steps
            mixed_precision: Use mixed precision training
            save_steps: Save checkpoint every N steps
            eval_steps: Evaluate every N steps
            logging_steps: Log metrics every N steps
            max_steps: Maximum training steps
            num_epochs: Number of epochs (ignored if max_steps set)
            warmup_steps: Number of warmup steps
            eval_metric: Metric to use for best model selection
            compute_generation_metrics: Whether to compute ROUGE/BLEU metrics
            tokenizer: Tokenizer for generation metrics
        """
        self.model = model.to(device)
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Training parameters
        self.gradient_clip = gradient_clip
        self.accumulation_steps = accumulation_steps
        self.mixed_precision = mixed_precision
        self.save_steps = save_steps
        self.eval_steps = eval_steps
        self.logging_steps = logging_steps
        self.max_steps = max_steps
        self.num_epochs = num_epochs
        self.warmup_steps = warmup_steps
        self.warmup_ratio = warmup_ratio
        self.eval_metric = eval_metric
        self.compute_generation_metrics = compute_generation_metrics
        self.tokenizer = tokenizer
        
        # Initialize evaluator if generation metrics requested
        if self.compute_generation_metrics:
            if tokenizer is None:
                raise ValueError("Tokenizer required for generation metrics")
            self.evaluator = EncoderEvaluator(
                encoder=self.model.encoder if hasattr(self.model, 'encoder') else None,
                full_model=self.model,
                tokenizer=tokenizer,
                device=device
            )
        
        # Initialize optimizer
        if optimizer is None:
            self.optimizer = self._create_optimizer()
        else:
            self.optimizer = optimizer
        
        # Initialize scheduler
        if scheduler is None:
            self.scheduler = self._create_scheduler()
        else:
            self.scheduler = scheduler
        
        # Mixed precision
        self.scaler = torch.cuda.amp.GradScaler() if mixed_precision else None
        
        # Logging
        self.log_wandb = log_wandb
        if log_wandb:
            self._init_wandb()
        
        # Training state
        self.global_step = 0
        self.current_epoch = 0
        self.best_metric = float('inf') if 'loss' in eval_metric else float('-inf')
        self.training_history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rate': []
        }
    
    def _create_optimizer(self) -> torch.optim.Optimizer:
        """Create default optimizer"""
        # Separate parameters for different learning rates
        encoder_params = []
        llm_params = []
        other_params = []
        
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            
            if 'eeg_encoder' in name:
                encoder_params.append(param)
            elif 'llm_decoder' in name:
                llm_params.append(param)
            else:
                other_params.append(param)
        
        # Different learning rates for different components
        param_groups = [
            {'params': encoder_params, 'lr': 1e-4},
            {'params': llm_params, 'lr': 5e-5},
            {'params': other_params, 'lr': 1e-4}
        ]
        
        return AdamW(param_groups, weight_decay=0.01, betas=(0.9, 0.999))
    
    def _create_scheduler(self) -> torch.optim.lr_scheduler._LRScheduler:
        """Create default scheduler"""
        total_steps = self.max_steps or len(self.train_dataloader) * self.num_epochs
        
        # Use warmup_ratio if provided, otherwise fall back to warmup_steps
        if self.warmup_ratio is not None:
            pct_start = min(1.0, max(0.0, self.warmup_ratio))
        else:
            # Ensure pct_start is between 0 and 1 when using fixed warmup_steps
            pct_start = min(1.0, max(0.0, self.warmup_steps / total_steps))
        
        return OneCycleLR(
            self.optimizer,
            max_lr=[1e-4, 5e-5, 1e-4],  # Match param groups
            total_steps=total_steps,
            pct_start=pct_start,
            anneal_strategy='cos'
        )
    
    def _init_wandb(self):
        """Initialize Weights & Biases logging"""
        wandb.init(
            project="brain-to-text",
            name=f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            config={
                'model_params': self.model.get_num_params(),
                'trainable_params': self.model.get_num_params(only_trainable=True),
                'batch_size': self.train_dataloader.batch_size,
                'accumulation_steps': self.accumulation_steps,
                'learning_rate': self.optimizer.param_groups[0]['lr'],
                'gradient_clip': self.gradient_clip,
                'mixed_precision': self.mixed_precision
            }
        )
    
    def train(self):
        """Main training loop"""
        self.model.train()
        
        # Determine total steps
        if self.max_steps:
            total_steps = self.max_steps
        elif self.num_epochs:
            total_steps = len(self.train_dataloader) * self.num_epochs
        else:
            raise ValueError("Either max_steps or num_epochs must be specified")
        
        # Training loop
        epoch_iterator = range(self.num_epochs) if self.num_epochs else range(1000)  # Large number
        
        for epoch in epoch_iterator:
            self.current_epoch = epoch
            epoch_loss = 0.0
            
            for step, batch in enumerate(tqdm(self.train_dataloader, desc=f"Epoch {epoch}")):
                # Move batch to device
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                # Forward pass
                with torch.cuda.amp.autocast(enabled=self.mixed_precision):
                    outputs = self.model(**batch)
                    loss = outputs['loss'] / self.accumulation_steps
                
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
                
                # Accumulate loss
                epoch_loss += loss.item() * self.accumulation_steps
                
                # Logging
                if self.global_step % self.logging_steps == 0:
                    avg_loss = epoch_loss / (step + 1)
                    self._log_metrics({
                        'train/loss': avg_loss,
                        'train/lm_loss': outputs.get('lm_loss', 0).item(),
                        'train/vq_loss': outputs.get('vq_loss', 0).item(),
                        'train/learning_rate': self.scheduler.get_last_lr()[0]
                    })
                
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
            epoch_metrics = {
                'epoch': epoch,
                'train_loss': epoch_loss / len(self.train_dataloader)
            }
            
            if self.val_dataloader:
                val_metrics = self.evaluate()
                epoch_metrics.update({f'val_{k}': v for k, v in val_metrics.items()})
            
            print(f"Epoch {epoch} completed: {epoch_metrics}")
    
    def evaluate(self) -> Dict[str, float]:
        """Evaluate model on validation set"""
        self.model.eval()
        
        total_loss = 0.0
        total_lm_loss = 0.0
        total_vq_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(self.val_dataloader, desc="Evaluating"):
                # Move batch to device
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                # Forward pass
                with torch.cuda.amp.autocast(enabled=self.mixed_precision):
                    outputs = self.model(**batch)
                
                total_loss += outputs['loss'].item()
                total_lm_loss += outputs.get('lm_loss', 0).item()
                total_vq_loss += outputs.get('vq_loss', 0).item()
                num_batches += 1
        
        self.model.train()
        
        metrics = {
            'loss': total_loss / num_batches,
            'lm_loss': total_lm_loss / num_batches,
            'vq_loss': total_vq_loss / num_batches
        }
        
        # Compute generation metrics if requested
        if self.compute_generation_metrics and hasattr(self, 'evaluator'):
            print("Computing generation metrics...")
            generation_metrics = self.evaluate_with_generation_metrics(limit_batches=5)
            metrics.update(generation_metrics)
        
        return metrics
    
    def evaluate_with_generation_metrics(self, limit_batches: Optional[int] = None) -> Dict[str, float]:
        """
        Extended evaluation with text generation metrics
        
        Args:
            limit_batches: Limit number of batches to evaluate (for speed)
            
        Returns:
            Dictionary of generation metrics
        """
        if not hasattr(self, 'evaluator'):
            return {}
        
        self.model.eval()
        
        all_predictions = []
        all_references = []
        batch_count = 0
        
        with torch.no_grad():
            for batch in tqdm(self.val_dataloader, desc="Computing generation metrics"):
                if limit_batches and batch_count >= limit_batches:
                    break
                
                # Move batch to device
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                # Get EEG inputs and reference texts
                eeg_inputs = batch.get('eeg', batch.get('input'))
                reference_texts = batch.get('text', batch.get('target_text', []))
                
                if eeg_inputs is None or not reference_texts:
                    continue
                
                # Evaluate batch
                batch_results = self.evaluator.evaluate_batch(eeg_inputs, reference_texts)
                all_predictions.extend(batch_results['predictions'])
                all_references.extend(reference_texts)
                
                batch_count += 1
        
        # Compute overall metrics
        if all_predictions and all_references:
            metrics = self.evaluator.metrics_evaluator.compute_all_metrics(
                all_predictions, all_references
            )
            
            # Prefix metrics with 'gen_' to distinguish from loss metrics
            return {f'gen_{k}': v for k, v in metrics.items()}
        
        self.model.train()
        return {}
    
    def _is_better_metric(self, metrics: Dict[str, float]) -> bool:
        """Check if current metrics are better than best"""
        current_metric = metrics.get(self.eval_metric, metrics['loss'])
        
        if 'loss' in self.eval_metric:
            is_better = current_metric < self.best_metric
        else:
            is_better = current_metric > self.best_metric
        
        if is_better:
            self.best_metric = current_metric
        
        return is_better
    
    def _log_metrics(self, metrics: Dict[str, float]):
        """Log metrics to wandb and console"""
        if self.log_wandb:
            wandb.log(metrics, step=self.global_step)
        
        # Update training history
        for key, value in metrics.items():
            if key in self.training_history:
                self.training_history[key].append(value)
    
    def save_checkpoint(self, name: str):
        """Save model checkpoint"""
        checkpoint_dir = self.output_dir / name
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model
        self.model.save_pretrained(checkpoint_dir)
        
        # Save training state
        torch.save({
            'global_step': self.global_step,
            'epoch': self.current_epoch,
            'best_metric': self.best_metric,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'training_history': self.training_history,
            'scaler_state_dict': self.scaler.state_dict() if self.scaler else None
        }, checkpoint_dir / 'training_state.pt')
        
        # Save config
        with open(checkpoint_dir / 'trainer_config.json', 'w') as f:
            json.dump({
                'gradient_clip': self.gradient_clip,
                'accumulation_steps': self.accumulation_steps,
                'mixed_precision': self.mixed_precision,
                'eval_metric': self.eval_metric
            }, f, indent=2)
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load checkpoint and resume training"""
        checkpoint_dir = Path(checkpoint_path)
        
        # Load model
        self.model = self.model.__class__.from_pretrained(checkpoint_dir)
        self.model.to(self.device)
        
        # Load training state
        state = torch.load(checkpoint_dir / 'training_state.pt')
        self.global_step = state['global_step']
        self.current_epoch = state['epoch']
        self.best_metric = state['best_metric']
        self.optimizer.load_state_dict(state['optimizer_state_dict'])
        self.scheduler.load_state_dict(state['scheduler_state_dict'])
        self.training_history = state['training_history']
        
        if self.scaler and state['scaler_state_dict']:
            self.scaler.load_state_dict(state['scaler_state_dict'])
