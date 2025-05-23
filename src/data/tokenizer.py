"""
Tokenizer for Brain-to-Text Model
"""
import json
from typing import List, Dict, Optional, Union
from pathlib import Path
import torch
from transformers import AutoTokenizer, PreTrainedTokenizer
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace


class BrainToTextTokenizer:
    """
    Custom tokenizer for Brain-to-Text model
    Handles both text tokens and special EEG tokens
    """
    
    def __init__(self,
                 vocab_size: int = 32000,
                 model_max_length: int = 512,
                 pad_token: str = "<pad>",
                 unk_token: str = "<unk>",
                 bos_token: str = "<s>",
                 eos_token: str = "</s>",
                 eeg_token: str = "<eeg>",
                 sep_token: str = "<sep>",
                 additional_special_tokens: Optional[List[str]] = None):
        """
        Initialize tokenizer
        
        Args:
            vocab_size: Size of vocabulary
            model_max_length: Maximum sequence length
            pad_token: Padding token
            unk_token: Unknown token
            bos_token: Beginning of sentence token
            eos_token: End of sentence token
            eeg_token: Special token for EEG data
            sep_token: Separator token between EEG and text
            additional_special_tokens: Additional special tokens
        """
        self.vocab_size = vocab_size
        self.model_max_length = model_max_length
        self.pad_token = pad_token
        self.unk_token = unk_token
        self.bos_token = bos_token
        self.eos_token = eos_token
        self.eeg_token = eeg_token
        self.sep_token = sep_token
        
        # Special tokens
        self.special_tokens = [
            pad_token, unk_token, bos_token, eos_token, 
            eeg_token, sep_token
        ]
        
        if additional_special_tokens:
            self.special_tokens.extend(additional_special_tokens)
        
        # Initialize base tokenizer (will be set by train or from_pretrained)
        self.tokenizer = None
        self._is_fast = False
        
    def train(self, texts: List[str], vocab_size: Optional[int] = None):
        """
        Train tokenizer on text corpus
        
        Args:
            texts: List of text strings for training
            vocab_size: Override default vocab size
        """
        if vocab_size is None:
            vocab_size = self.vocab_size
        
        # Create BPE tokenizer
        tokenizer = Tokenizer(BPE(unk_token=self.unk_token))
        tokenizer.pre_tokenizer = Whitespace()
        
        # Create trainer
        trainer = BpeTrainer(
            vocab_size=vocab_size,
            special_tokens=self.special_tokens,
            show_progress=True
        )
        
        # Train tokenizer
        tokenizer.train_from_iterator(texts, trainer)
        
        # Set up post-processing
        tokenizer.post_processor = self._create_post_processor()
        
        self.tokenizer = tokenizer
        self._is_fast = True
        
        # Build vocab mapping
        self._build_vocab()
    
    def _create_post_processor(self):
        """Create post-processor for adding special tokens"""
        from tokenizers.processors import TemplateProcessing
        
        return TemplateProcessing(
            single=f"{self.bos_token} $A {self.eos_token}",
            pair=f"{self.bos_token} $A {self.sep_token} $B {self.eos_token}",
            special_tokens=[
                (self.bos_token, self.tokenizer.token_to_id(self.bos_token)),
                (self.eos_token, self.tokenizer.token_to_id(self.eos_token)),
                (self.sep_token, self.tokenizer.token_to_id(self.sep_token)),
            ]
        )
    
    def _build_vocab(self):
        """Build vocabulary mappings"""
        self.vocab = {}
        self.ids_to_tokens = {}
        
        if self._is_fast:
            # Fast tokenizer
            vocab = self.tokenizer.get_vocab()
            self.vocab = vocab
            self.ids_to_tokens = {v: k for k, v in vocab.items()}
        else:
            # HuggingFace tokenizer
            self.vocab = self.tokenizer.get_vocab()
            self.ids_to_tokens = {v: k for k, v in self.vocab.items()}
        
        # Store special token IDs
        self.pad_token_id = self.vocab.get(self.pad_token, 0)
        self.unk_token_id = self.vocab.get(self.unk_token, 1)
        self.bos_token_id = self.vocab.get(self.bos_token, 2)
        self.eos_token_id = self.vocab.get(self.eos_token, 3)
        self.eeg_token_id = self.vocab.get(self.eeg_token, 4)
        self.sep_token_id = self.vocab.get(self.sep_token, 5)
    
    def encode(self, 
               text: Union[str, List[str]], 
               add_special_tokens: bool = True,
               padding: Union[bool, str] = False,
               truncation: bool = False,
               max_length: Optional[int] = None,
               return_tensors: Optional[str] = None) -> Union[List[int], Dict[str, torch.Tensor]]:
        """
        Encode text to token IDs
        
        Args:
            text: Text string or list of strings
            add_special_tokens: Whether to add special tokens
            padding: Padding strategy
            truncation: Whether to truncate
            max_length: Maximum length
            return_tensors: Return type ('pt' for PyTorch tensors)
            
        Returns:
            Token IDs or dict with input_ids and attention_mask
        """
        if max_length is None:
            max_length = self.model_max_length
        
        if isinstance(text, str):
            texts = [text]
            single_text = True
        else:
            texts = text
            single_text = False
        
        # Encode texts
        if self._is_fast:
            encodings = self.tokenizer.encode_batch(texts)
            input_ids = [enc.ids for enc in encodings]
        else:
            input_ids = [self.tokenizer.encode(t, add_special_tokens=add_special_tokens) 
                        for t in texts]
        
        # Truncate if needed
        if truncation:
            input_ids = [ids[:max_length] for ids in input_ids]
        
        # Pad if needed
        if padding:
            max_len = max(len(ids) for ids in input_ids)
            if padding == 'max_length':
                max_len = max_length
            
            attention_mask = []
            for i, ids in enumerate(input_ids):
                pad_len = max_len - len(ids)
                attention_mask.append([1] * len(ids) + [0] * pad_len)
                input_ids[i] = ids + [self.pad_token_id] * pad_len
        
        # Return appropriate format
        if return_tensors == 'pt':
            result = {
                'input_ids': torch.tensor(input_ids),
            }
            if padding:
                result['attention_mask'] = torch.tensor(attention_mask)
            
            if single_text:
                result = {k: v.squeeze(0) for k, v in result.items()}
            
            return result
        else:
            if single_text:
                return input_ids[0]
            return input_ids
    
    def decode(self, 
               token_ids: Union[List[int], torch.Tensor],
               skip_special_tokens: bool = True) -> str:
        """
        Decode token IDs to text
        
        Args:
            token_ids: Token IDs
            skip_special_tokens: Whether to skip special tokens
            
        Returns:
            Decoded text
        """
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
        
        if self._is_fast:
            return self.tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)
        else:
            return self.tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)
    
    def batch_decode(self,
                    token_ids: Union[List[List[int]], torch.Tensor],
                    skip_special_tokens: bool = True) -> List[str]:
        """
        Decode batch of token IDs to texts
        
        Args:
            token_ids: Batch of token IDs
            skip_special_tokens: Whether to skip special tokens
            
        Returns:
            List of decoded texts
        """
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
        
        return [self.decode(ids, skip_special_tokens) for ids in token_ids]
    
    def save_pretrained(self, save_directory: str):
        """Save tokenizer to directory"""
        save_dir = Path(save_directory)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Save tokenizer
        if self._is_fast:
            self.tokenizer.save(str(save_dir / "tokenizer.json"))
        else:
            self.tokenizer.save_pretrained(save_directory)
        
        # Save config
        config = {
            'vocab_size': self.vocab_size,
            'model_max_length': self.model_max_length,
            'pad_token': self.pad_token,
            'unk_token': self.unk_token,
            'bos_token': self.bos_token,
            'eos_token': self.eos_token,
            'eeg_token': self.eeg_token,
            'sep_token': self.sep_token,
            'special_tokens': self.special_tokens,
            'is_fast': self._is_fast
        }
        
        with open(save_dir / "tokenizer_config.json", 'w') as f:
            json.dump(config, f, indent=2)
    
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, **kwargs):
        """Load tokenizer from pretrained"""
        path = Path(pretrained_model_name_or_path)
        
        # Load config
        with open(path / "tokenizer_config.json", 'r') as f:
            config = json.load(f)
        
        # Create tokenizer instance
        tokenizer = cls(
            vocab_size=config['vocab_size'],
            model_max_length=config['model_max_length'],
            pad_token=config['pad_token'],
            unk_token=config['unk_token'],
            bos_token=config['bos_token'],
            eos_token=config['eos_token'],
            eeg_token=config['eeg_token'],
            sep_token=config['sep_token']
        )
        
        # Load tokenizer
        if config['is_fast']:
            from tokenizers import Tokenizer
            tokenizer.tokenizer = Tokenizer.from_file(str(path / "tokenizer.json"))
            tokenizer._is_fast = True
        else:
            tokenizer.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path)
            tokenizer._is_fast = False
        
        # Build vocab
        tokenizer._build_vocab()
        
        return tokenizer
    
    def __call__(self, *args, **kwargs):
        """Make tokenizer callable like HuggingFace tokenizers"""
        return self.encode(*args, **kwargs)
    
    @classmethod
    def from_huggingface(cls, model_name: str, **kwargs):
        """Create tokenizer from HuggingFace model"""
        # Load HuggingFace tokenizer
        hf_tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Create our tokenizer
        tokenizer = cls(**kwargs)
        tokenizer.tokenizer = hf_tokenizer
        tokenizer._is_fast = hasattr(hf_tokenizer, 'backend_tokenizer')
        tokenizer._build_vocab()
        
        return tokenizer
