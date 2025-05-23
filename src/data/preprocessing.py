"""
MEG/EEG Preprocessing utilities for Brain-to-Text Model
"""
import numpy as np
import torch
from typing import Optional, Tuple, List, Union
import mne
from scipy import signal
from scipy.stats import zscore
import warnings
warnings.filterwarnings('ignore')


class MEGPreprocessor:
    """
    Preprocessing pipeline for MEG/EEG signals
    """
    
    def __init__(self,
                 sampling_rate: int = 1000,
                 lowpass: float = 100.0,
                 highpass: float = 0.1,
                 notch_freqs: List[float] = [50.0, 100.0],
                 reference: str = 'average',
                 normalize: bool = True,
                 artifact_rejection: bool = True,
                 ica_components: int = 20):
        """
        Initialize preprocessor
        
        Args:
            sampling_rate: Target sampling rate
            lowpass: Low-pass filter frequency
            highpass: High-pass filter frequency
            notch_freqs: Frequencies for notch filtering (e.g., line noise)
            reference: Reference method ('average', 'REST', or None)
            normalize: Whether to normalize the data
            artifact_rejection: Whether to apply artifact rejection
            ica_components: Number of ICA components for artifact removal
        """
        self.sampling_rate = sampling_rate
        self.lowpass = lowpass
        self.highpass = highpass
        self.notch_freqs = notch_freqs
        self.reference = reference
        self.normalize = normalize
        self.artifact_rejection = artifact_rejection
        self.ica_components = ica_components
    
    def process_raw(self, raw: mne.io.Raw) -> mne.io.Raw:
        """
        Apply full preprocessing pipeline to raw MEG/EEG data
        
        Args:
            raw: MNE Raw object
            
        Returns:
            Preprocessed Raw object
        """
        # Make a copy to avoid modifying original
        raw = raw.copy()
        
        # Pick MEG/EEG channels
        raw.pick_types(meg=True, eeg=True, stim=False, eog=False, ecg=False)
        
        # Resample if needed
        if raw.info['sfreq'] != self.sampling_rate:
            raw.resample(self.sampling_rate)
        
        # Apply filters
        if self.highpass is not None and self.lowpass is not None:
            raw.filter(l_freq=self.highpass, h_freq=self.lowpass, 
                      method='fir', phase='zero-double', verbose=False)
        
        # Notch filter for line noise
        if self.notch_freqs:
            raw.notch_filter(freqs=self.notch_freqs, verbose=False)
        
        # Re-reference
        if self.reference == 'average':
            raw.set_eeg_reference('average', projection=False, verbose=False)
        elif self.reference == 'REST':
            # REST reference requires forward model
            pass  # Implement if needed
        
        # Artifact rejection with ICA
        if self.artifact_rejection and self.ica_components > 0:
            raw = self._apply_ica(raw)
        
        return raw
    
    def _apply_ica(self, raw: mne.io.Raw) -> mne.io.Raw:
        """Apply ICA for artifact removal"""
        from mne.preprocessing import ICA
        
        # Fit ICA
        ica = ICA(n_components=self.ica_components, 
                  method='fastica', 
                  random_state=42,
                  verbose=False)
        
        # Filter data for ICA (1-100 Hz is typical)
        raw_filt = raw.copy().filter(l_freq=1.0, h_freq=100.0, verbose=False)
        ica.fit(raw_filt)
        
        # Find and exclude EOG/ECG components automatically
        # This is a simplified version - in practice, you'd want more sophisticated detection
        eog_indices, eog_scores = ica.find_bads_eog(raw, threshold=3.0)
        ecg_indices, ecg_scores = ica.find_bads_ecg(raw, threshold=3.0)
        
        ica.exclude = list(set(eog_indices + ecg_indices))
        
        # Apply ICA
        raw = ica.apply(raw)
        
        return raw
    
    def process_array(self, data: np.ndarray, 
                     info: Optional[mne.Info] = None) -> np.ndarray:
        """
        Process numpy array of MEG/EEG data
        
        Args:
            data: Data array [channels, time]
            info: MNE info object (optional)
            
        Returns:
            Processed data array
        """
        # Apply temporal filters
        if self.highpass is not None or self.lowpass is not None:
            nyquist = self.sampling_rate / 2
            
            if self.highpass is not None and self.lowpass is not None:
                # Band-pass filter
                sos = signal.butter(5, [self.highpass/nyquist, self.lowpass/nyquist], 
                                   btype='band', output='sos')
            elif self.highpass is not None:
                # High-pass filter
                sos = signal.butter(5, self.highpass/nyquist, 
                                   btype='high', output='sos')
            else:
                # Low-pass filter
                sos = signal.butter(5, self.lowpass/nyquist, 
                                   btype='low', output='sos')
            
            data = signal.sosfiltfilt(sos, data, axis=1)
        
        # Notch filter
        if self.notch_freqs:
            for freq in self.notch_freqs:
                b, a = signal.iirnotch(freq, Q=30, fs=self.sampling_rate)
                data = signal.filtfilt(b, a, data, axis=1)
        
        # Normalize
        if self.normalize:
            data = self._normalize_data(data)
        
        return data
    
    def _normalize_data(self, data: np.ndarray) -> np.ndarray:
        """
        Normalize data using various methods
        
        Args:
            data: Data array [channels, time] or [batch, channels, time]
            
        Returns:
            Normalized data
        """
        # Z-score normalization per channel
        if data.ndim == 2:
            # Single sample
            data = zscore(data, axis=1)
        elif data.ndim == 3:
            # Batch
            for i in range(data.shape[0]):
                data[i] = zscore(data[i], axis=1)
        
        # Clip extreme values
        data = np.clip(data, -5, 5)
        
        return data
    
    def augment_data(self, data: np.ndarray, 
                    noise_level: float = 0.1,
                    time_shift_range: int = 50,
                    channel_dropout_rate: float = 0.1) -> np.ndarray:
        """
        Apply data augmentation for training
        
        Args:
            data: Data array [channels, time]
            noise_level: Gaussian noise standard deviation
            time_shift_range: Maximum time shift in samples
            channel_dropout_rate: Probability of dropping a channel
            
        Returns:
            Augmented data
        """
        data = data.copy()
        
        # Add Gaussian noise
        if noise_level > 0:
            noise = np.random.normal(0, noise_level, data.shape)
            data += noise
        
        # Random time shift
        if time_shift_range > 0:
            shift = np.random.randint(-time_shift_range, time_shift_range)
            if shift > 0:
                data = np.pad(data, ((0, 0), (shift, 0)), mode='constant')[:, :-shift]
            elif shift < 0:
                data = np.pad(data, ((0, 0), (0, -shift)), mode='constant')[:, -shift:]
        
        # Channel dropout
        if channel_dropout_rate > 0:
            dropout_mask = np.random.random(data.shape[0]) > channel_dropout_rate
            data[~dropout_mask] = 0
        
        return data
    
    def extract_features(self, data: np.ndarray) -> dict:
        """
        Extract various features from MEG/EEG data
        
        Args:
            data: Data array [channels, time]
            
        Returns:
            Dictionary of features
        """
        features = {}
        
        # Power spectral density
        freqs, psd = signal.welch(data, fs=self.sampling_rate, nperseg=1024)
        
        # Band powers
        bands = {
            'delta': (0.5, 4),
            'theta': (4, 8),
            'alpha': (8, 13),
            'beta': (13, 30),
            'gamma': (30, 100)
        }
        
        for band_name, (low, high) in bands.items():
            band_mask = (freqs >= low) & (freqs <= high)
            features[f'{band_name}_power'] = np.mean(psd[:, band_mask], axis=1)
        
        # Statistical features
        features['mean'] = np.mean(data, axis=1)
        features['std'] = np.std(data, axis=1)
        features['skewness'] = self._compute_skewness(data)
        features['kurtosis'] = self._compute_kurtosis(data)
        
        # Temporal features
        features['peak_to_peak'] = np.ptp(data, axis=1)
        features['zero_crossings'] = self._count_zero_crossings(data)
        
        return features
    
    def _compute_skewness(self, data: np.ndarray) -> np.ndarray:
        """Compute skewness for each channel"""
        from scipy.stats import skew
        return skew(data, axis=1)
    
    def _compute_kurtosis(self, data: np.ndarray) -> np.ndarray:
        """Compute kurtosis for each channel"""
        from scipy.stats import kurtosis
        return kurtosis(data, axis=1)
    
    def _count_zero_crossings(self, data: np.ndarray) -> np.ndarray:
        """Count zero crossings for each channel"""
        zero_crossings = np.zeros(data.shape[0])
        for ch in range(data.shape[0]):
            zero_crossings[ch] = np.sum(np.diff(np.sign(data[ch])) != 0)
        return zero_crossings
    
    def create_spectrogram(self, data: np.ndarray, 
                          nperseg: int = 256,
                          noverlap: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Create spectrogram representation
        
        Args:
            data: Data array [channels, time]
            nperseg: Length of each segment
            noverlap: Number of points to overlap
            
        Returns:
            frequencies, times, spectrogram [channels, frequencies, time]
        """
        if noverlap is None:
            noverlap = nperseg // 2
        
        # Compute spectrogram for first channel to get dimensions
        f, t, _ = signal.spectrogram(data[0], fs=self.sampling_rate, 
                                     nperseg=nperseg, noverlap=noverlap)
        
        # Initialize output array
        Sxx = np.zeros((data.shape[0], len(f), len(t)))
        
        # Compute for all channels
        for ch in range(data.shape[0]):
            _, _, Sxx[ch] = signal.spectrogram(data[ch], fs=self.sampling_rate,
                                              nperseg=nperseg, noverlap=noverlap)
        
        return f, t, Sxx


class BatchPreprocessor:
    """
    Batch preprocessing for PyTorch tensors
    """
    
    def __init__(self, preprocessor: MEGPreprocessor):
        self.preprocessor = preprocessor
    
    def __call__(self, batch: torch.Tensor) -> torch.Tensor:
        """
        Process a batch of MEG/EEG data
        
        Args:
            batch: Tensor [batch, channels, time]
            
        Returns:
            Processed tensor
        """
        # Convert to numpy
        batch_np = batch.numpy()
        
        # Process each sample
        processed = np.zeros_like(batch_np)
        for i in range(batch_np.shape[0]):
            processed[i] = self.preprocessor.process_array(batch_np[i])
        
        # Convert back to tensor
        return torch.from_numpy(processed).float()
