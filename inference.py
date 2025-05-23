"""
Inference script for Brain-to-Text Model
"""
import argparse
import torch
import numpy as np
from pathlib import Path
from typing import Optional, List, Dict
import json
from tqdm import tqdm

from src.models.full_model import BrainToTextModel
from src.data.preprocessing import MEGPreprocessor
from src.data.tokenizer import BrainToTextTokenizer
from transformers import AutoTokenizer


class BrainToTextInference:
    """
    Inference pipeline for Brain-to-Text model
    """
    
    def __init__(self,
                 model_path: str,
                 device: str = 'cuda',
                 tokenizer_path: Optional[str] = None,
                 preprocessor_config: Optional[dict] = None):
        """
        Initialize inference pipeline
        
        Args:
            model_path: Path to trained model checkpoint
            device: Device to run inference on
            tokenizer_path: Path to tokenizer (if different from model)
            preprocessor_config: Configuration for MEG preprocessing
        """
        self.device = torch.device(device)
        
        # Load model
        print(f"Loading model from {model_path}")
        self.model = BrainToTextModel.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()
        
        # Load tokenizer
        if tokenizer_path:
            print(f"Loading tokenizer from {tokenizer_path}")
            self.tokenizer = BrainToTextTokenizer.from_pretrained(tokenizer_path)
        else:
            # Try to load from model directory
            tokenizer_path = Path(model_path) / "tokenizer"
            if tokenizer_path.exists():
                self.tokenizer = BrainToTextTokenizer.from_pretrained(str(tokenizer_path))
            else:
                # Use default tokenizer
                print("Using default LLaMA tokenizer")
                self.tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
        
        # Initialize preprocessor
        if preprocessor_config is None:
            preprocessor_config = {
                'sampling_rate': 1000,
                'lowpass': 100.0,
                'highpass': 0.1,
                'normalize': True
            }
        self.preprocessor = MEGPreprocessor(**preprocessor_config)
    
    @torch.no_grad()
    def generate_text(self,
                     meg_signals: torch.Tensor,
                     max_length: int = 256,
                     temperature: float = 0.8,
                     top_k: int = 50,
                     top_p: float = 0.95,
                     num_beams: int = 1,
                     do_sample: bool = True) -> Dict[str, any]:
        """
        Generate text from MEG signals
        
        Args:
            meg_signals: MEG signals tensor [batch_size, n_channels, time_steps]
            max_length: Maximum length of generated text
            temperature: Sampling temperature
            top_k: Top-k sampling parameter
            top_p: Top-p (nucleus) sampling parameter
            num_beams: Number of beams for beam search
            do_sample: Whether to use sampling
            
        Returns:
            Dictionary with generated text and additional information
        """
        # Move to device
        meg_signals = meg_signals.to(self.device)
        
        # Generate text
        outputs = self.model.generate(
            meg_signals=meg_signals,
            max_length=max_length,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            num_beams=num_beams,
            do_sample=do_sample,
            pad_token_id=self.tokenizer.pad_token_id if hasattr(self.tokenizer, 'pad_token_id') else 0,
            eos_token_id=self.tokenizer.eos_token_id if hasattr(self.tokenizer, 'eos_token_id') else 2
        )
        
        # Decode generated text
        generated_ids = outputs['generated_ids']
        generated_texts = []
        
        for ids in generated_ids:
            text = self.tokenizer.decode(ids, skip_special_tokens=True)
            generated_texts.append(text)
        
        # Get additional information
        result = {
            'generated_texts': generated_texts,
            'generated_ids': generated_ids.cpu().numpy(),
            'eeg_indices': outputs.get('eeg_indices', None)
        }
        
        if 'eeg_indices' in outputs and outputs['eeg_indices'] is not None:
            result['eeg_indices'] = outputs['eeg_indices'].cpu().numpy()
        
        return result
    
    def process_meg_file(self, 
                        meg_file_path: str,
                        segment_length: float = 2.0,
                        overlap: float = 0.5,
                        **generation_kwargs) -> List[Dict]:
        """
        Process MEG file and generate text for segments
        
        Args:
            meg_file_path: Path to MEG file (.fif format)
            segment_length: Length of segments in seconds
            overlap: Overlap between segments (0-1)
            **generation_kwargs: Arguments for text generation
            
        Returns:
            List of results for each segment
        """
        import mne
        
        # Load MEG data
        print(f"Loading MEG data from {meg_file_path}")
        raw = mne.io.read_raw_fif(meg_file_path, preload=True, verbose=False)
        
        # Preprocess
        raw = self.preprocessor.process_raw(raw)
        
        # Get data
        data = raw.get_data()
        sampling_rate = int(raw.info['sfreq'])
        
        # Segment data
        segment_samples = int(segment_length * sampling_rate)
        overlap_samples = int(segment_samples * overlap)
        step_samples = segment_samples - overlap_samples
        
        results = []
        start_idx = 0
        
        while start_idx + segment_samples <= data.shape[1]:
            # Extract segment
            segment = data[:, start_idx:start_idx + segment_samples]
            
            # Convert to tensor and add batch dimension
            meg_tensor = torch.from_numpy(segment).float().unsqueeze(0)
            
            # Generate text
            result = self.generate_text(meg_tensor, **generation_kwargs)
            
            # Add timing information
            result['start_time'] = start_idx / sampling_rate
            result['end_time'] = (start_idx + segment_samples) / sampling_rate
            
            results.append(result)
            start_idx += step_samples
        
        return results
    
    def process_batch(self,
                     meg_batch: np.ndarray,
                     **generation_kwargs) -> Dict[str, any]:
        """
        Process a batch of MEG segments
        
        Args:
            meg_batch: Batch of MEG segments [batch_size, n_channels, time_steps]
            **generation_kwargs: Arguments for text generation
            
        Returns:
            Dictionary with results
        """
        # Preprocess each segment
        processed_batch = np.zeros_like(meg_batch)
        for i in range(meg_batch.shape[0]):
            processed_batch[i] = self.preprocessor.process_array(meg_batch[i])
        
        # Convert to tensor
        meg_tensor = torch.from_numpy(processed_batch).float()
        
        # Generate text
        return self.generate_text(meg_tensor, **generation_kwargs)


def main():
    parser = argparse.ArgumentParser(description='Run inference with Brain-to-Text Model')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--input_file', type=str, required=True,
                       help='Path to MEG file (.fif) or numpy file')
    parser.add_argument('--output_file', type=str, default='output.json',
                       help='Path to save results')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use')
    parser.add_argument('--segment_length', type=float, default=2.0,
                       help='Segment length in seconds')
    parser.add_argument('--overlap', type=float, default=0.5,
                       help='Overlap between segments')
    parser.add_argument('--max_length', type=int, default=256,
                       help='Maximum length of generated text')
    parser.add_argument('--temperature', type=float, default=0.8,
                       help='Sampling temperature')
    parser.add_argument('--top_k', type=int, default=50,
                       help='Top-k sampling')
    parser.add_argument('--top_p', type=float, default=0.95,
                       help='Top-p sampling')
    parser.add_argument('--num_beams', type=int, default=1,
                       help='Number of beams for beam search')
    parser.add_argument('--do_sample', action='store_true',
                       help='Use sampling (vs greedy decoding)')
    
    args = parser.parse_args()
    
    # Initialize inference pipeline
    pipeline = BrainToTextInference(
        model_path=args.model_path,
        device=args.device
    )
    
    # Generation arguments
    gen_kwargs = {
        'max_length': args.max_length,
        'temperature': args.temperature,
        'top_k': args.top_k,
        'top_p': args.top_p,
        'num_beams': args.num_beams,
        'do_sample': args.do_sample
    }
    
    # Process input
    input_path = Path(args.input_file)
    
    if input_path.suffix == '.fif':
        # Process MEG file
        results = pipeline.process_meg_file(
            args.input_file,
            segment_length=args.segment_length,
            overlap=args.overlap,
            **gen_kwargs
        )
    elif input_path.suffix in ['.npy', '.npz']:
        # Process numpy array
        if input_path.suffix == '.npy':
            meg_data = np.load(args.input_file)
        else:
            meg_data = np.load(args.input_file)['data']
        
        # Add batch dimension if needed
        if meg_data.ndim == 2:
            meg_data = meg_data[np.newaxis, :]
        
        results = pipeline.process_batch(meg_data, **gen_kwargs)
        results = [results]  # Wrap in list for consistency
    else:
        raise ValueError(f"Unsupported input format: {input_path.suffix}")
    
    # Save results
    output_data = {
        'input_file': args.input_file,
        'model_path': args.model_path,
        'generation_params': gen_kwargs,
        'results': results
    }
    
    with open(args.output_file, 'w') as f:
        json.dump(output_data, f, indent=2, default=str)
    
    print(f"Results saved to {args.output_file}")
    
    # Print sample results
    print("\nSample results:")
    for i, result in enumerate(results[:3]):  # Show first 3
        print(f"\nSegment {i+1}:")
        if 'start_time' in result:
            print(f"Time: {result['start_time']:.2f}s - {result['end_time']:.2f}s")
        print(f"Generated text: {result['generated_texts'][0]}")


if __name__ == "__main__":
    main()
