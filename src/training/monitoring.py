"""
Training monitoring and early stopping utilities
"""
import torch
import numpy as np
from typing import Dict, List, Optional, Callable
from pathlib import Path
import json
import wandb
from collections import deque


class EarlyStopping:
    """
    Early stopping handler with patience and monitoring
    """
    
    def __init__(self,
                 patience: int = 10,
                 min_delta: float = 0.001,
                 mode: str = 'min',
                 baseline: Optional[float] = None,
                 restore_best_weights: bool = True):
        """
        Initialize early stopping
        
        Args:
            patience: Number of epochs with no improvement after which training will be stopped
            min_delta: Minimum change in the monitored quantity to qualify as an improvement
            mode: One of {'min', 'max'}. In 'min' mode, training will stop when the monitored
                  quantity stops decreasing; in 'max' mode it will stop when it stops increasing
            baseline: Baseline value for the monitored quantity
            restore_best_weights: Whether to restore model weights from the best epoch
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.baseline = baseline
        self.restore_best_weights = restore_best_weights
        
        self.best_score = None
        self.best_epoch = None
        self.best_weights = None
        self.wait_count = 0
        self.stopped_epoch = None
        
        if mode == 'min':
            self.monitor_op = np.less
            self.min_delta *= -1
        else:
            self.monitor_op = np.greater
    
    def __call__(self, score: float, model: Optional[torch.nn.Module] = None, 
                 epoch: Optional[int] = None) -> bool:
        """
        Check if training should stop
        
        Args:
            score: Current score to monitor
            model: Model to save best weights from
            epoch: Current epoch number
            
        Returns:
            True if training should stop, False otherwise
        """
        if self.baseline is not None:
            score = self.monitor_op(score, self.baseline)
        
        if self.best_score is None:
            self.best_score = score
            self.best_epoch = epoch
            if self.restore_best_weights and model is not None:
                self.best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        elif self.monitor_op(score - self.min_delta, self.best_score):
            self.best_score = score
            self.best_epoch = epoch
            self.wait_count = 0
            if self.restore_best_weights and model is not None:
                self.best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            self.wait_count += 1
            if self.wait_count >= self.patience:
                self.stopped_epoch = epoch
                if self.restore_best_weights and model is not None and self.best_weights is not None:
                    model.load_state_dict(self.best_weights)
                return True
        
        return False
    
    def get_status(self) -> Dict:
        """Get current early stopping status"""
        return {
            'best_score': self.best_score,
            'best_epoch': self.best_epoch,
            'wait_count': self.wait_count,
            'stopped_epoch': self.stopped_epoch
        }


class TrainingMonitor:
    """
    Comprehensive training monitor with metrics tracking and visualization
    """
    
    def __init__(self,
                 output_dir: str,
                 metrics_to_track: List[str] = None,
                 window_size: int = 100,
                 log_wandb: bool = True):
        """
        Initialize training monitor
        
        Args:
            output_dir: Directory to save monitoring data
            metrics_to_track: List of metric names to track
            window_size: Window size for moving averages
            log_wandb: Whether to log to wandb
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.metrics_to_track = metrics_to_track or [
            'loss', 'lm_loss', 'vq_loss', 'reconstruction_loss',
            'diversity_loss', 'perplexity_loss', 'rvq_utilization',
            'avg_perplexity', 'temperature'
        ]
        
        self.window_size = window_size
        self.log_wandb = log_wandb
        
        # Initialize tracking
        self.metrics_history = {metric: [] for metric in self.metrics_to_track}
        self.moving_averages = {metric: deque(maxlen=window_size) for metric in self.metrics_to_track}
        self.best_metrics = {}
        self.step_count = 0
    
    def update(self, metrics: Dict[str, float], step: Optional[int] = None):
        """
        Update metrics
        
        Args:
            metrics: Dictionary of metric values
            step: Current training step
        """
        if step is not None:
            self.step_count = step
        else:
            self.step_count += 1
        
        # Update histories
        for metric_name, value in metrics.items():
            if metric_name in self.metrics_history:
                self.metrics_history[metric_name].append((self.step_count, value))
                self.moving_averages[metric_name].append(value)
                
                # Track best metrics
                if metric_name not in self.best_metrics:
                    self.best_metrics[metric_name] = {
                        'value': value,
                        'step': self.step_count
                    }
                else:
                    # Update best for loss metrics (lower is better)
                    if 'loss' in metric_name and value < self.best_metrics[metric_name]['value']:
                        self.best_metrics[metric_name] = {
                            'value': value,
                            'step': self.step_count
                        }
                    # Update best for other metrics (higher is better)
                    elif 'loss' not in metric_name and value > self.best_metrics[metric_name]['value']:
                        self.best_metrics[metric_name] = {
                            'value': value,
                            'step': self.step_count
                        }
        
        # Log to wandb
        if self.log_wandb:
            try:
                wandb.log(metrics, step=self.step_count)
            except Exception as e:
                # Silently continue if wandb logging fails
                pass
    
    def get_moving_average(self, metric_name: str) -> Optional[float]:
        """Get moving average for a metric"""
        if metric_name in self.moving_averages and len(self.moving_averages[metric_name]) > 0:
            return np.mean(list(self.moving_averages[metric_name]))
        return None
    
    def check_improvement(self, metric_name: str, window: int = 1000) -> bool:
        """
        Check if metric is improving over a window
        
        Args:
            metric_name: Metric to check
            window: Number of steps to look back
            
        Returns:
            True if metric is improving, False otherwise
        """
        if metric_name not in self.metrics_history:
            return False
        
        history = self.metrics_history[metric_name]
        if len(history) < window:
            return True  # Not enough data yet
        
        # Get recent and past values
        recent_values = [v for _, v in history[-window//2:]]
        past_values = [v for _, v in history[-window:-window//2]]
        
        if not recent_values or not past_values:
            return True
        
        recent_avg = np.mean(recent_values)
        past_avg = np.mean(past_values)
        
        # Check improvement based on metric type
        if 'loss' in metric_name:
            return recent_avg < past_avg
        else:
            return recent_avg > past_avg
    
    def save_checkpoint(self, model: torch.nn.Module, optimizer: torch.optim.Optimizer,
                       scheduler: Optional[object] = None, epoch: Optional[int] = None,
                       is_best: bool = False):
        """Save training checkpoint"""
        checkpoint = {
            'step': self.step_count,
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metrics_history': self.metrics_history,
            'best_metrics': self.best_metrics
        }
        
        if scheduler is not None:
            checkpoint['scheduler_state_dict'] = scheduler.state_dict()
        
        # Save regular checkpoint
        checkpoint_path = self.output_dir / f'checkpoint_step_{self.step_count}.pt'
        torch.save(checkpoint, checkpoint_path)
        
        # Save best checkpoint
        if is_best:
            best_path = self.output_dir / 'best_checkpoint.pt'
            torch.save(checkpoint, best_path)
        
        # Save metrics summary
        self.save_metrics_summary()
    
    def save_metrics_summary(self):
        """Save metrics summary to JSON"""
        summary = {
            'current_step': self.step_count,
            'best_metrics': self.best_metrics,
            'current_values': {
                metric: self.get_moving_average(metric)
                for metric in self.metrics_to_track
            }
        }
        
        with open(self.output_dir / 'metrics_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
    
    def generate_report(self) -> str:
        """Generate training report"""
        report = []
        report.append(f"Training Report - Step {self.step_count}")
        report.append("=" * 50)
        
        # Current metrics
        report.append("\nCurrent Metrics (Moving Average):")
        for metric in self.metrics_to_track:
            avg = self.get_moving_average(metric)
            if avg is not None:
                report.append(f"  {metric}: {avg:.6f}")
        
        # Best metrics
        report.append("\nBest Metrics:")
        for metric, info in self.best_metrics.items():
            report.append(f"  {metric}: {info['value']:.6f} (step {info['step']})")
        
        # Improvement status
        report.append("\nImprovement Status (last 1000 steps):")
        for metric in ['loss', 'rvq_utilization', 'avg_perplexity']:
            if metric in self.metrics_to_track:
                improving = self.check_improvement(metric)
                status = "✓ Improving" if improving else "✗ Not improving"
                report.append(f"  {metric}: {status}")
        
        return "\n".join(report)


class RVQMonitor:
    """
    Specialized monitor for RVQ module metrics
    """
    
    def __init__(self, num_quantizers: int, codebook_size: int):
        """
        Initialize RVQ monitor
        
        Args:
            num_quantizers: Number of quantizers
            codebook_size: Size of each codebook
        """
        self.num_quantizers = num_quantizers
        self.codebook_size = codebook_size
        self.usage_history = []
        self.perplexity_history = []
        self.temperature_history = []
    
    def update(self, rvq_stats: Dict, step: int):
        """Update RVQ statistics"""
        # Extract statistics
        utilization_rates = []
        perplexities = []
        
        for i in range(self.num_quantizers):
            key = f'quantizer_{i}'
            if key in rvq_stats:
                stats = rvq_stats[key]
                utilization_rates.append(stats.get('usage_rate', 0.0))
                perplexities.append(stats.get('perplexity', 0.0))
        
        # Store history
        self.usage_history.append((step, utilization_rates))
        self.perplexity_history.append((step, perplexities))
        
        if 'overall' in rvq_stats and 'temperature' in rvq_stats['overall']:
            self.temperature_history.append((step, rvq_stats['overall']['temperature']))
    
    def check_codebook_health(self) -> Dict[str, bool]:
        """Check if codebooks are healthy"""
        if not self.usage_history:
            return {'healthy': True, 'issues': []}
        
        latest_usage = self.usage_history[-1][1]
        latest_perplexity = self.perplexity_history[-1][1] if self.perplexity_history else []
        
        issues = []
        
        # Check for dead codebooks
        for i, usage in enumerate(latest_usage):
            if usage < 0.01:  # Less than 1% usage
                issues.append(f"Codebook {i} is nearly dead (usage: {usage:.2%})")
        
        # Check for low perplexity
        for i, perplexity in enumerate(latest_perplexity):
            if perplexity < self.codebook_size * 0.1:  # Less than 10% of codebook size
                issues.append(f"Codebook {i} has low perplexity ({perplexity:.1f})")
        
        return {
            'healthy': len(issues) == 0,
            'issues': issues,
            'avg_usage': np.mean(latest_usage) if latest_usage else 0.0,
            'avg_perplexity': np.mean(latest_perplexity) if latest_perplexity else 0.0
        }