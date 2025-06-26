"""
Script to evaluate the encoder module with ROUGE and BLEU metrics
"""
import torch
import yaml
import argparse
from pathlib import Path
from transformers import AutoTokenizer
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.models.eeg_encoder import EEGEncoder
from src.models.full_model import BrainToTextModel
from src.data.meg_dataset import MEGDataset
from src.evaluation.encoder_evaluator import EncoderEvaluator
from torch.utils.data import DataLoader


def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_evaluation_dataloader(config, data_path):
    """Create dataloader for evaluation"""
    # Update data path
    config['data']['data_root'] = data_path
    
    # Create dataset
    dataset = MEGDataset(
        data_root=config['data']['data_root'],
        subjects=config['data'].get('subjects'),
        sessions=config['data'].get('sessions'),
        tasks=config['data'].get('tasks'),
        sampling_rate=config['data']['sampling_rate'],
        segment_length=config['data']['segment_length'],
        overlap=config['data']['overlap'],
        lowpass=config['data']['lowpass'],
        highpass=config['data']['highpass'],
        normalize=config['data']['normalize'],
        cache_dir=config['data']['cache_dir'],
        split='val'  # Use validation split for evaluation
    )
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['data']['num_workers'],
        pin_memory=True
    )
    
    return dataloader


def main():
    parser = argparse.ArgumentParser(description='Evaluate encoder with generation metrics')
    parser.add_argument('--config', type=str, default='configs/default_config.yaml',
                        help='Path to configuration file')
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to MASC-MEG dataset')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to model checkpoint')
    parser.add_argument('--output_dir', type=str, default='./evaluation_results',
                        help='Directory to save evaluation results')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to run evaluation on')
    parser.add_argument('--limit_batches', type=int, default=None,
                        help='Limit number of batches to evaluate')
    
    args = parser.parse_args()
    
    # Load configuration
    print(f"Loading configuration from {args.config}")
    config = load_config(args.config)
    
    # Update device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Initialize tokenizer
    print("Initializing tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(config['tokenizer']['model_name'])
    
    # Initialize model
    print("Initializing model...")
    model = BrainToTextModel(config)
    
    # Load checkpoint if provided
    if args.checkpoint:
        print(f"Loading checkpoint from {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
    
    model = model.to(device)
    model.eval()
    
    # Get encoder
    encoder = model.encoder
    
    # Initialize evaluator
    print("Initializing evaluator...")
    evaluator = EncoderEvaluator(
        encoder=encoder,
        full_model=model,
        tokenizer=tokenizer,
        device=str(device)
    )
    
    # Update generation config if specified
    if 'evaluation' in config and 'generation_params' in config['evaluation']:
        evaluator.set_generation_config(config['evaluation']['generation_params'])
    
    # Create evaluation dataloader
    print(f"Loading data from {args.data_path}")
    try:
        dataloader = create_evaluation_dataloader(config, args.data_path)
        print(f"Created dataloader with {len(dataloader)} batches")
    except Exception as e:
        print(f"Error creating dataloader: {e}")
        print("Please ensure the data path is correct and the dataset is properly formatted")
        return
    
    # Run evaluation
    print("\nStarting evaluation...")
    try:
        # Limit batches if specified
        if args.limit_batches:
            print(f"Limiting evaluation to {args.limit_batches} batches")
            # Create limited dataloader
            limited_data = []
            for i, batch in enumerate(dataloader):
                if i >= args.limit_batches:
                    break
                limited_data.append(batch)
            
            # Evaluate on limited data
            results = evaluator.evaluate_on_dataset(
                limited_data,
                save_results=True,
                output_dir=args.output_dir
            )
        else:
            # Full evaluation
            results = evaluator.evaluate_on_dataset(
                dataloader,
                save_results=True,
                output_dir=args.output_dir
            )
        
        # Print results
        print("\n" + "="*50)
        print("EVALUATION RESULTS")
        print("="*50)
        
        if 'corpus_level' in results:
            print("\nCorpus-Level Metrics:")
            for metric, value in results['corpus_level'].items():
                if isinstance(value, float):
                    print(f"  {metric.upper()}: {value:.4f}")
        
        if 'mean_values' in results:
            print("\nAverage Metrics:")
            for metric, value in results['mean_values'].items():
                print(f"  {metric}: {value:.4f}")
        
        if 'dataset_stats' in results:
            print("\nDataset Statistics:")
            for stat, value in results['dataset_stats'].items():
                print(f"  {stat}: {value}")
        
        print(f"\nDetailed results saved to: {args.output_dir}")
        
    except Exception as e:
        print(f"Error during evaluation: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()