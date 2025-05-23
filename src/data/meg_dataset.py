"""
MEG-MASC Dataset loader for Brain-to-Text Model
"""
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
import os
import mne
from pathlib import Path
import json
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)


class MEGDataset(Dataset):
    """
    Dataset class for MEG-MASC data
    Handles loading MEG recordings with corresponding text annotations
    """
    
    def __init__(self,
                 data_root: str,
                 subjects: Optional[List[str]] = None,
                 sessions: Optional[List[str]] = None,
                 tasks: Optional[List[str]] = None,
                 sampling_rate: int = 1000,
                 segment_length: float = 2.0,  # seconds
                 overlap: float = 0.5,  # overlap between segments
                 preprocess: bool = True,
                 max_text_length: int = 256,
                 cache_dir: Optional[str] = None):
        """
        Initialize MEG dataset
        
        Args:
            data_root: Root directory of MEG-MASC dataset
            subjects: List of subject IDs to include (None for all)
            sessions: List of session IDs to include (None for all)
            tasks: List of task IDs to include (None for all)
            sampling_rate: Target sampling rate for MEG data
            segment_length: Length of each segment in seconds
            overlap: Overlap between segments (0-1)
            preprocess: Whether to apply preprocessing
            max_text_length: Maximum text sequence length
            cache_dir: Directory for caching processed data
        """
        self.data_root = Path(data_root)
        self.subjects = subjects
        self.sessions = sessions
        self.tasks = tasks
        self.sampling_rate = sampling_rate
        self.segment_length = segment_length
        self.overlap = overlap
        self.preprocess = preprocess
        self.max_text_length = max_text_length
        self.cache_dir = Path(cache_dir) if cache_dir else None
        
        # Task mapping from README
        self.task_names = {
            0: 'lw1',
            1: 'cable_spool_fort',
            2: 'easy_money',
            3: 'The_Black_Widow'
        }
        
        # Load file paths and metadata
        self.samples = self._load_samples()
        
    def _load_samples(self) -> List[Dict]:
        """Load all available MEG samples with metadata"""
        samples = []
        
        # Iterate through subjects
        subject_dirs = [d for d in self.data_root.iterdir() if d.is_dir() and d.name.startswith('sub-')]
        
        if self.subjects:
            subject_dirs = [d for d in subject_dirs if d.name in self.subjects]
        
        for subject_dir in subject_dirs:
            subject_id = subject_dir.name
            
            # Iterate through sessions
            session_dirs = [d for d in subject_dir.iterdir() if d.is_dir() and d.name.startswith('ses-')]
            
            if self.sessions:
                session_dirs = [d for d in session_dirs if d.name in self.sessions]
            
            for session_dir in session_dirs:
                session_id = session_dir.name
                meg_dir = session_dir / 'meg'
                
                if not meg_dir.exists():
                    continue
                
                # Find MEG files
                meg_files = list(meg_dir.glob('*_meg.fif'))
                
                for meg_file in meg_files:
                    # Extract task info from filename
                    parts = meg_file.stem.split('_')
                    task_idx = None
                    
                    for part in parts:
                        if part.startswith('task-'):
                            try:
                                task_idx = int(part.replace('task-', ''))
                            except:
                                continue
                    
                    if task_idx is None:
                        continue
                    
                    if self.tasks and task_idx not in self.tasks:
                        continue
                    
                    # Find corresponding events file
                    events_file = meg_file.with_name(meg_file.stem.replace('_meg', '_events') + '.tsv')
                    
                    if not events_file.exists():
                        continue
                    
                    # Create sample entry
                    sample = {
                        'subject_id': subject_id,
                        'session_id': session_id,
                        'task_idx': task_idx,
                        'task_name': self.task_names.get(task_idx, f'task-{task_idx}'),
                        'meg_file': str(meg_file),
                        'events_file': str(events_file),
                    }
                    
                    samples.append(sample)
        
        return samples
    
    def _load_meg_data(self, meg_file: str) -> Tuple[np.ndarray, int]:
        """Load MEG data from FIF file"""
        # Check cache first
        if self.cache_dir:
            cache_file = self.cache_dir / f"{Path(meg_file).stem}_processed.npy"
            if cache_file.exists():
                data = np.load(cache_file)
                return data, self.sampling_rate
        
        # Load raw MEG data
        raw = mne.io.read_raw_fif(meg_file, preload=True, verbose=False)
        
        # Pick MEG channels only
        raw.pick_types(meg=True, eeg=False, stim=False, eog=False, ecg=False)
        
        # Resample if needed
        if raw.info['sfreq'] != self.sampling_rate:
            raw.resample(self.sampling_rate)
        
        # Apply preprocessing if requested
        if self.preprocess:
            # Band-pass filter
            raw.filter(l_freq=0.1, h_freq=100, verbose=False)
            
            # Notch filter for line noise
            raw.notch_filter(freqs=[50, 100], verbose=False)
        
        # Get data
        data = raw.get_data()
        
        # Save to cache
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            np.save(cache_file, data)
        
        return data, int(raw.info['sfreq'])
    
    def _load_events(self, events_file: str) -> pd.DataFrame:
        """Load events from TSV file"""
        events = pd.read_csv(events_file, sep='\t')
        
        # Parse trial_type column if it contains dict strings
        if 'trial_type' in events.columns:
            def parse_trial_type(x):
                if isinstance(x, str) and x.startswith('{'):
                    try:
                        return eval(x)
                    except:
                        return {}
                return x
            
            events['trial_type_dict'] = events['trial_type'].apply(parse_trial_type)
        
        return events
    
    def _extract_segments(self, data: np.ndarray, events: pd.DataFrame, 
                         sampling_rate: int) -> List[Dict]:
        """Extract segments from continuous MEG data based on events"""
        segments = []
        
        # Calculate segment parameters
        segment_samples = int(self.segment_length * sampling_rate)
        overlap_samples = int(segment_samples * self.overlap)
        step_samples = segment_samples - overlap_samples
        
        # Get word events
        word_events = events[events['trial_type'].str.contains('word', case=False, na=False)]
        
        if len(word_events) == 0:
            # No word events, create segments from continuous data
            n_samples = data.shape[1]
            start_idx = 0
            
            while start_idx + segment_samples <= n_samples:
                segment = {
                    'data': data[:, start_idx:start_idx + segment_samples],
                    'start_time': start_idx / sampling_rate,
                    'end_time': (start_idx + segment_samples) / sampling_rate,
                    'text': '',  # No text available
                    'words': []
                }
                segments.append(segment)
                start_idx += step_samples
        else:
            # Group events by temporal proximity
            current_segment_words = []
            current_start_sample = None
            
            for idx, event in word_events.iterrows():
                event_sample = int(event['sample'])
                
                if current_start_sample is None:
                    current_start_sample = event_sample
                    current_segment_words = [event]
                elif event_sample - current_start_sample < segment_samples:
                    current_segment_words.append(event)
                else:
                    # Create segment from accumulated words
                    if current_segment_words:
                        start_sample = max(0, current_start_sample)
                        end_sample = min(data.shape[1], start_sample + segment_samples)
                        
                        # Extract words from trial_type
                        words = []
                        for word_event in current_segment_words:
                            if isinstance(word_event['trial_type_dict'], dict):
                                word = word_event['trial_type_dict'].get('word', '')
                                if word:
                                    words.append(word)
                        
                        segment = {
                            'data': data[:, start_sample:end_sample],
                            'start_time': start_sample / sampling_rate,
                            'end_time': end_sample / sampling_rate,
                            'text': ' '.join(words),
                            'words': words
                        }
                        segments.append(segment)
                    
                    # Start new segment
                    current_start_sample = event_sample
                    current_segment_words = [event]
            
            # Don't forget the last segment
            if current_segment_words and current_start_sample is not None:
                start_sample = max(0, current_start_sample)
                end_sample = min(data.shape[1], start_sample + segment_samples)
                
                words = []
                for word_event in current_segment_words:
                    if isinstance(word_event['trial_type_dict'], dict):
                        word = word_event['trial_type_dict'].get('word', '')
                        if word:
                            words.append(word)
                
                segment = {
                    'data': data[:, start_sample:end_sample],
                    'start_time': start_sample / sampling_rate,
                    'end_time': end_sample / sampling_rate,
                    'text': ' '.join(words),
                    'words': words
                }
                segments.append(segment)
        
        return segments
    
    def __len__(self) -> int:
        # For simplicity, return number of files
        # In practice, we'd want to pre-compute total segments
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        
        # Load MEG data
        meg_data, sampling_rate = self._load_meg_data(sample['meg_file'])
        
        # Load events
        events = self._load_events(sample['events_file'])
        
        # Extract segments
        segments = self._extract_segments(meg_data, events, sampling_rate)
        
        if len(segments) == 0:
            # Return empty sample if no segments found
            return {
                'meg_data': torch.zeros(208, int(self.segment_length * self.sampling_rate)),
                'text': '',
                'subject_id': sample['subject_id'],
                'task_name': sample['task_name'],
                'segment_idx': 0
            }
        
        # For now, return first segment
        # In practice, you might want to return all segments or random segment
        segment = segments[0]
        
        # Convert to tensors
        meg_tensor = torch.from_numpy(segment['data']).float()
        
        # Pad or truncate to fixed length
        target_length = int(self.segment_length * self.sampling_rate)
        if meg_tensor.shape[1] < target_length:
            # Pad
            pad_length = target_length - meg_tensor.shape[1]
            meg_tensor = torch.nn.functional.pad(meg_tensor, (0, pad_length))
        elif meg_tensor.shape[1] > target_length:
            # Truncate
            meg_tensor = meg_tensor[:, :target_length]
        
        return {
            'meg_data': meg_tensor,
            'text': segment['text'],
            'subject_id': sample['subject_id'],
            'task_name': sample['task_name'],
            'segment_idx': 0
        }


class MEGDataCollator:
    """
    Data collator for MEG-text pairs
    Handles batching and padding
    """
    
    def __init__(self, tokenizer, max_length: int = 256):
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __call__(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        # Stack MEG data
        meg_data = torch.stack([item['meg_data'] for item in batch])
        
        # Tokenize text
        texts = [item['text'] for item in batch]
        
        if self.tokenizer is not None:
            encoded = self.tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors='pt'
            )
            
            return {
                'meg_signals': meg_data,
                'input_ids': encoded['input_ids'],
                'attention_mask': encoded['attention_mask'],
                'labels': encoded['input_ids'].clone()
            }
        else:
            # Return without tokenization
            return {
                'meg_signals': meg_data,
                'texts': texts
            }
