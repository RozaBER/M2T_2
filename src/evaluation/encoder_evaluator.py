import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm
import json
import os
from datetime import datetime

from .metrics import EncoderEvaluationMetrics
from .utils import aggregate_metrics, generate_evaluation_report


class EncoderEvaluator:
    """Comprehensive evaluation pipeline for encoder module"""
    
    def __init__(self, encoder: nn.Module, full_model: nn.Module, tokenizer, device: str = 'cuda'):
        """
        Initialize the encoder evaluator
        
        Args:
            encoder: EEG encoder module
            full_model: Complete brain-to-text model
            tokenizer: Text tokenizer
            device: Device to run evaluation on
        """
        self.encoder = encoder
        self.full_model = full_model
        self.tokenizer = tokenizer
        self.device = device
        self.metrics_evaluator = EncoderEvaluationMetrics()
        
        # Default generation configuration
        self.generation_config = {
            'max_length': 128,
            'num_beams': 4,
            'temperature': 0.7,
            'top_p': 0.9,
            'do_sample': False,
            'early_stopping': True
        }
    
    def set_generation_config(self, config: Dict):
        """Update generation configuration"""
        self.generation_config.update(config)
    
    def evaluate_on_dataset(self, dataloader: DataLoader, 
                          save_results: bool = True,
                          output_dir: str = './evaluation_results') -> Dict[str, float]:
        """
        Evaluate encoder on entire dataset
        
        Args:
            dataloader: DataLoader containing evaluation data
            save_results: Whether to save evaluation results
            output_dir: Directory to save results
            
        Returns:
            Dictionary of aggregated metrics
        """
        self.encoder.eval()
        self.full_model.eval()
        
        all_predictions = []
        all_references = []
        batch_metrics = []
        
        print("Evaluating encoder on dataset...")
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating batches"):
                # Extract data from batch
                eeg_inputs = batch['eeg'].to(self.device)
                reference_texts = batch['text']
                
                # Evaluate batch
                batch_result = self.evaluate_batch(eeg_inputs, reference_texts)
                batch_metrics.append(batch_result['metrics'])
                
                all_predictions.extend(batch_result['predictions'])
                all_references.extend(reference_texts)
        
        # Aggregate metrics across all batches
        final_metrics = aggregate_metrics(batch_metrics)
        
        # Compute corpus-level metrics
        corpus_metrics = self.metrics_evaluator.compute_all_metrics(
            all_predictions, all_references
        )
        final_metrics['corpus_level'] = corpus_metrics
        
        # Add dataset statistics
        final_metrics['dataset_stats'] = {
            'total_samples': len(all_predictions),
            'total_batches': len(batch_metrics),
            'avg_prediction_length': sum(len(p.split()) for p in all_predictions) / len(all_predictions),
            'avg_reference_length': sum(len(r.split()) for r in all_references) / len(all_references)
        }
        
        if save_results:
            self._save_results(final_metrics, all_predictions, all_references, output_dir)
        
        return final_metrics
    
    def evaluate_batch(self, eeg_inputs: torch.Tensor, 
                      reference_texts: List[str]) -> Dict:
        """
        Evaluate a single batch
        
        Args:
            eeg_inputs: Batch of EEG inputs
            reference_texts: Corresponding reference texts
            
        Returns:
            Dictionary containing predictions and metrics
        """
        # Use encoder's evaluation method
        metrics = self.encoder.evaluate_generation_quality(
            eeg_inputs,
            reference_texts,
            self.full_model,
            self.tokenizer,
            self.metrics_evaluator,
            self.generation_config
        )
        
        # Generate predictions for returning
        with torch.no_grad():
            if hasattr(self.full_model, 'generate'):
                generated_ids = self.full_model.generate(
                    eeg_inputs,
                    **self.generation_config
                )
            else:
                # Fallback to encoder + decoder pipeline
                encoded_features = self.encoder(eeg_inputs)
                generated_ids = self._generate_from_features(encoded_features)
            
            predictions = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            predictions = [pred.strip() for pred in predictions]
        
        return {
            'predictions': predictions,
            'metrics': metrics
        }
    
    def evaluate_encoder_representations(self, dataloader: DataLoader) -> Dict:
        """
        Analyze encoder output representations
        
        Args:
            dataloader: DataLoader containing evaluation data
            
        Returns:
            Dictionary of representation analysis metrics
        """
        self.encoder.eval()
        
        all_features = []
        all_labels = []
        
        print("Analyzing encoder representations...")
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Extracting features"):
                eeg_inputs = batch['eeg'].to(self.device)
                
                # Get encoder outputs
                features = self.encoder(eeg_inputs)
                
                # Store features and labels
                all_features.append(features.cpu())
                if 'label' in batch:
                    all_labels.extend(batch['label'])
        
        # Concatenate all features
        all_features = torch.cat(all_features, dim=0)
        
        # Compute representation metrics
        representation_metrics = {
            'feature_stats': {
                'mean_norm': all_features.norm(dim=-1).mean().item(),
                'std_norm': all_features.norm(dim=-1).std().item(),
                'mean_activation': all_features.mean().item(),
                'std_activation': all_features.std().item(),
                'sparsity': (all_features == 0).float().mean().item()
            },
            'feature_shape': list(all_features.shape)
        }
        
        # Compute similarity metrics if labels available
        if all_labels:
            representation_metrics['similarity_analysis'] = self._compute_similarity_metrics(
                all_features, all_labels
            )
        
        return representation_metrics
    
    def _generate_from_features(self, features: torch.Tensor) -> torch.Tensor:
        """Generate text from encoder features (fallback method)"""
        # This is a placeholder - implement based on your model architecture
        # For now, return dummy ids
        batch_size = features.shape[0]
        return torch.randint(0, 1000, (batch_size, 50))
    
    def _compute_similarity_metrics(self, features: torch.Tensor, 
                                  labels: List) -> Dict:
        """Compute similarity metrics for encoded representations"""
        # Compute cosine similarity between samples with same/different labels
        from sklearn.metrics.pairwise import cosine_similarity
        import numpy as np
        
        features_np = features.mean(dim=1).numpy()  # Average over sequence length
        similarity_matrix = cosine_similarity(features_np)
        
        # Compute intra-class and inter-class similarities
        unique_labels = list(set(labels))
        label_to_idx = {label: i for i, label in enumerate(unique_labels)}
        
        intra_class_sims = []
        inter_class_sims = []
        
        for i in range(len(features_np)):
            for j in range(i + 1, len(features_np)):
                sim = similarity_matrix[i, j]
                if labels[i] == labels[j]:
                    intra_class_sims.append(sim)
                else:
                    inter_class_sims.append(sim)
        
        return {
            'mean_intra_class_similarity': np.mean(intra_class_sims) if intra_class_sims else 0,
            'mean_inter_class_similarity': np.mean(inter_class_sims) if inter_class_sims else 0,
            'similarity_ratio': (np.mean(intra_class_sims) / np.mean(inter_class_sims)) if inter_class_sims else 0
        }
    
    def _save_results(self, metrics: Dict, predictions: List[str], 
                     references: List[str], output_dir: str):
        """Save evaluation results to disk"""
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save metrics
        metrics_file = os.path.join(output_dir, f'metrics_{timestamp}.json')
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        # Save predictions and references
        predictions_file = os.path.join(output_dir, f'predictions_{timestamp}.txt')
        with open(predictions_file, 'w') as f:
            for pred, ref in zip(predictions, references):
                f.write(f"Prediction: {pred}\n")
                f.write(f"Reference: {ref}\n")
                f.write("-" * 80 + "\n")
        
        # Generate evaluation report
        report_file = os.path.join(output_dir, f'report_{timestamp}.md')
        generate_evaluation_report(metrics, report_file)
        
        print(f"Results saved to {output_dir}")