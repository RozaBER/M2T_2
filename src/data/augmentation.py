"""
Data augmentation techniques for MEG/EEG signals
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, List


class MEGAugmentation(nn.Module):
    """
    Data augmentation module for MEG/EEG signals
    Applies various augmentations to improve model robustness
    """
    
    def __init__(self,
                 time_shift_range: float = 0.1,  # ±100ms
                 channel_dropout_prob: float = 0.1,
                 gaussian_noise_std: float = 0.01,
                 scaling_range: Tuple[float, float] = (0.9, 1.1),
                 frequency_masking_param: int = 20,
                 time_masking_param: int = 50,
                 apply_prob: float = 0.5):
        """
        Initialize MEG augmentation module
        
        Args:
            time_shift_range: Maximum time shift in seconds
            channel_dropout_prob: Probability of dropping channels
            gaussian_noise_std: Standard deviation of Gaussian noise
            scaling_range: Range for signal scaling
            frequency_masking_param: Maximum frequency channels to mask
            time_masking_param: Maximum time steps to mask
            apply_prob: Probability of applying each augmentation
        """
        super().__init__()
        
        self.time_shift_range = time_shift_range
        self.channel_dropout_prob = channel_dropout_prob
        self.gaussian_noise_std = gaussian_noise_std
        self.scaling_range = scaling_range
        self.frequency_masking_param = frequency_masking_param
        self.time_masking_param = time_masking_param
        self.apply_prob = apply_prob
    
    def forward(self, x: torch.Tensor, sampling_rate: int = 1000) -> torch.Tensor:
        """
        Apply augmentations to MEG signal
        
        Args:
            x: Input signal [batch, channels, time]
            sampling_rate: Sampling rate in Hz
            
        Returns:
            Augmented signal
        """
        if not self.training:
            return x
        
        batch_size = x.size(0)
        device = x.device
        
        # Apply each augmentation with probability
        for i in range(batch_size):
            # Time shift
            if torch.rand(1).item() < self.apply_prob:
                x[i] = self._time_shift(x[i], sampling_rate)
            
            # Channel dropout
            if torch.rand(1).item() < self.apply_prob:
                x[i] = self._channel_dropout(x[i])
            
            # Gaussian noise
            if torch.rand(1).item() < self.apply_prob:
                x[i] = self._add_gaussian_noise(x[i])
            
            # Signal scaling
            if torch.rand(1).item() < self.apply_prob:
                x[i] = self._scale_signal(x[i])
            
            # Frequency masking
            if torch.rand(1).item() < self.apply_prob:
                x[i] = self._frequency_masking(x[i])
            
            # Time masking
            if torch.rand(1).item() < self.apply_prob:
                x[i] = self._time_masking(x[i])
        
        return x
    
    def _time_shift(self, x: torch.Tensor, sampling_rate: int) -> torch.Tensor:
        """Apply random time shift"""
        max_shift_samples = int(self.time_shift_range * sampling_rate)
        shift = torch.randint(-max_shift_samples, max_shift_samples + 1, (1,)).item()
        
        if shift > 0:
            # Shift right (pad left)
            x = F.pad(x, (shift, 0), mode='constant', value=0)[:, shift:]
        elif shift < 0:
            # Shift left (pad right)
            x = F.pad(x, (0, -shift), mode='constant', value=0)[:, :shift]
        
        return x
    
    def _channel_dropout(self, x: torch.Tensor) -> torch.Tensor:
        """Randomly drop channels"""
        n_channels = x.size(0)
        dropout_mask = torch.rand(n_channels, 1, device=x.device) > self.channel_dropout_prob
        return x * dropout_mask
    
    def _add_gaussian_noise(self, x: torch.Tensor) -> torch.Tensor:
        """Add Gaussian noise"""
        noise = torch.randn_like(x) * self.gaussian_noise_std
        return x + noise
    
    def _scale_signal(self, x: torch.Tensor) -> torch.Tensor:
        """Random signal scaling"""
        scale = torch.empty(1).uniform_(*self.scaling_range).to(x.device)
        return x * scale
    
    def _frequency_masking(self, x: torch.Tensor) -> torch.Tensor:
        """Mask random frequency bands"""
        # Apply FFT
        x_fft = torch.fft.rfft(x, dim=-1)
        
        # Random frequency mask
        freq_size = x_fft.size(-1)
        if self.frequency_masking_param > 0 and freq_size > self.frequency_masking_param:
            mask_size = torch.randint(0, self.frequency_masking_param, (1,)).item()
            if mask_size > 0:
                mask_start = torch.randint(0, freq_size - mask_size, (1,)).item()
                x_fft[:, mask_start:mask_start + mask_size] = 0
        
        # Apply inverse FFT
        x = torch.fft.irfft(x_fft, n=x.size(-1), dim=-1)
        return x
    
    def _time_masking(self, x: torch.Tensor) -> torch.Tensor:
        """Mask random time segments"""
        time_size = x.size(-1)
        if self.time_masking_param > 0 and time_size > self.time_masking_param:
            mask_size = torch.randint(0, self.time_masking_param, (1,)).item()
            if mask_size > 0:
                mask_start = torch.randint(0, time_size - mask_size, (1,)).item()
                x[:, mask_start:mask_start + mask_size] = 0
        return x


class SegmentAugmentation(nn.Module):
    """
    Augmentation for MEG segments with text alignment preservation
    """
    
    def __init__(self,
                 segment_shuffle_prob: float = 0.1,
                 segment_reverse_prob: float = 0.05,
                 mixup_alpha: float = 0.2,
                 cutmix_prob: float = 0.1):
        """
        Initialize segment augmentation
        
        Args:
            segment_shuffle_prob: Probability of shuffling segments
            segment_reverse_prob: Probability of reversing segments
            mixup_alpha: Alpha parameter for mixup
            cutmix_prob: Probability of applying cutmix
        """
        super().__init__()
        
        self.segment_shuffle_prob = segment_shuffle_prob
        self.segment_reverse_prob = segment_reverse_prob
        self.mixup_alpha = mixup_alpha
        self.cutmix_prob = cutmix_prob
    
    def mixup(self, x: torch.Tensor, y: torch.Tensor, 
              alpha: Optional[float] = None) -> Tuple[torch.Tensor, torch.Tensor, float]:
        """
        Apply mixup augmentation
        
        Args:
            x: Input signals [batch, channels, time]
            y: Target labels or embeddings
            alpha: Mixup parameter
            
        Returns:
            Mixed inputs, mixed targets, mixing weight
        """
        if alpha is None:
            alpha = self.mixup_alpha
        
        batch_size = x.size(0)
        if batch_size < 2:
            return x, y, 1.0
        
        # Sample lambda from Beta distribution
        lam = np.random.beta(alpha, alpha)
        
        # Random permutation
        index = torch.randperm(batch_size).to(x.device)
        
        # Mix inputs
        mixed_x = lam * x + (1 - lam) * x[index]
        
        # Mix targets if they are continuous (e.g., embeddings)
        if y.dtype in [torch.float16, torch.float32, torch.float64]:
            mixed_y = lam * y + (1 - lam) * y[index]
        else:
            # For discrete labels, return both
            mixed_y = (y, y[index], lam)
        
        return mixed_x, mixed_y, lam
    
    def cutmix(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, float]:
        """
        Apply CutMix augmentation
        
        Args:
            x: Input signals [batch, channels, time]
            y: Target labels or embeddings
            
        Returns:
            Mixed inputs, mixed targets, mixing weight
        """
        batch_size = x.size(0)
        if batch_size < 2 or torch.rand(1).item() > self.cutmix_prob:
            return x, y, 1.0
        
        # Sample lambda
        lam = np.random.beta(1.0, 1.0)
        
        # Random permutation
        index = torch.randperm(batch_size).to(x.device)
        
        # Get cut size
        time_size = x.size(-1)
        cut_size = int(time_size * (1 - lam))
        
        if cut_size > 0:
            # Random cut position
            cut_start = torch.randint(0, time_size - cut_size + 1, (1,)).item()
            
            # Apply cut
            x[:, :, cut_start:cut_start + cut_size] = x[index, :, cut_start:cut_start + cut_size]
        
        # Adjust lambda based on actual cut ratio
        lam = 1 - (cut_size / time_size)
        
        # Mix targets
        if y.dtype in [torch.float16, torch.float32, torch.float64]:
            mixed_y = lam * y + (1 - lam) * y[index]
        else:
            mixed_y = (y, y[index], lam)
        
        return x, mixed_y, lam


class AugmentedMEGDataset(torch.utils.data.Dataset):
    """
    Wrapper dataset that applies augmentation to MEG data
    """
    
    def __init__(self,
                 base_dataset: torch.utils.data.Dataset,
                 augmentation: MEGAugmentation,
                 segment_augmentation: Optional[SegmentAugmentation] = None):
        """
        Initialize augmented dataset
        
        Args:
            base_dataset: Original MEG dataset
            augmentation: MEG augmentation module
            segment_augmentation: Optional segment augmentation
        """
        self.base_dataset = base_dataset
        self.augmentation = augmentation
        self.segment_augmentation = segment_augmentation
    
    def __len__(self):
        return len(self.base_dataset)
    
    def __getitem__(self, idx):
        # Get base sample
        sample = self.base_dataset[idx]
        
        # Apply MEG augmentation
        if 'meg_data' in sample and isinstance(sample['meg_data'], torch.Tensor):
            sample['meg_data'] = self.augmentation(sample['meg_data'].unsqueeze(0)).squeeze(0)
        
        return sample