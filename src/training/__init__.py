"""
Training modules for Brain-to-Text Model
"""
from .trainer import BrainToTextTrainer
from .multi_phase_trainer import MultiPhaseTrainer
from .distillation import DistillationTrainer

__all__ = ['BrainToTextTrainer', 'MultiPhaseTrainer', 'DistillationTrainer']
