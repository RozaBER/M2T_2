#!/usr/bin/env python3
"""
Comprehensive evaluation script for Brain-to-Text model
Evaluates both encoder performance and full model generation quality
"""

import argparse
import torch
import json
import numpy as np
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend
import matplotlib.pyplot as plt
import seaborn as sns

from train import load_config, setup_dataset
from src.models.full_model import BrainToTextModel
from src.evaluation.encoder_evaluator import EncoderEvaluator
from src.evaluation.metrics import EncoderEvaluationMetrics


class ModelEvaluator:
    def __init__(self, model, tokenizer, device='cuda'):
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.device = device
        self.model.eval()
        self.metrics_evaluator = EncoderEvaluationMetrics()
        
    def evaluate_encoder_reconstruction(self, dataloader):
        """Evaluate encoder's ability to reconstruct MEG signals through RVQ"""
        print("\n=== Evaluating Encoder Reconstruction Quality ===")
        
        reconstruction_errors = []
        vq_losses = []
        codebook_usage = {i: np.zeros(self.model.rvq.codebook_size) 
                         for i in range(self.model.rvq.num_quantizers)}
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating reconstruction"):
                meg_signals = batch['eeg_signals'].to(self.device)
                
                # Encode and quantize
                encoded = self.model.eeg_encoder(meg_signals)
                quantized, indices, aux_losses = self.model.rvq(encoded)
                
                # Calculate reconstruction error
                recon_error = torch.mean((encoded - quantized) ** 2).item()
                reconstruction_errors.append(recon_error)
                
                # Track VQ losses
                if isinstance(aux_losses, dict):
                    vq_losses.append(aux_losses.get('total_vq_loss', 0).item())
                
                # Track codebook usage
                indices_np = indices.cpu().numpy()
                for i in range(indices.shape[-1]):
                    unique, counts = np.unique(indices_np[..., i], return_counts=True)
                    for idx, count in zip(unique, counts):
                        codebook_usage[i][idx] += count
        
        # Calculate statistics
        avg_recon_error = np.mean(reconstruction_errors)
        avg_vq_loss = np.mean(vq_losses) if vq_losses else 0
        
        # Calculate codebook utilization
        codebook_utilization = {}
        for i, usage in codebook_usage.items():
            utilized = np.sum(usage > 0)
            utilization_rate = utilized / len(usage)
            codebook_utilization[f'codebook_{i}'] = {
                'utilized_codes': int(utilized),
                'total_codes': len(usage),
                'utilization_rate': float(utilization_rate)
            }
        
        return {
            'avg_reconstruction_error': float(avg_recon_error),
            'avg_vq_loss': float(avg_vq_loss),
            'codebook_utilization': codebook_utilization
        }
    
    def evaluate_generation_quality(self, dataloader, max_samples=50):
        """Evaluate full model's text generation quality"""
        print("\n=== Evaluating Text Generation Quality ===")
        
        predictions = []
        references = []
        generation_losses = []
        
        # Generation parameters (only use supported parameters)
        gen_params = {
            'max_length': 128,
            'temperature': 0.7,
            'top_p': 0.9,
            'do_sample': False,  # Use greedy decoding for consistent evaluation
            'top_k': 50
        }
        
        with torch.no_grad():
            for i, batch in enumerate(tqdm(dataloader, desc="Generating text")):
                if i >= max_samples:
                    break
                    
                meg_signals = batch['eeg_signals'].to(self.device)
                attention_mask = batch.get('attention_mask', None)
                labels = batch.get('labels', batch.get('text_ids', None))
                
                # Generate text
                generated_ids = self.model.generate(
                    meg_signals,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    **gen_params
                )
                
                # Decode predictions and references
                for j in range(generated_ids.shape[0]):
                    pred_text = self.tokenizer.decode(generated_ids[j], skip_special_tokens=True)
                    predictions.append(pred_text)
                    
                    if labels is not None:
                        # Find where the actual text starts (after EEG tokens)
                        label_ids = labels[j].cpu().numpy()
                        text_start = np.where(label_ids >= 0)[0]
                        if len(text_start) > 0:
                            text_ids = label_ids[text_start[0]:]
                            ref_text = self.tokenizer.decode(text_ids, skip_special_tokens=True)
                            references.append(ref_text)
                        else:
                            references.append("")
                
                # Calculate generation loss if labels available
                if labels is not None:
                    input_ids = batch.get('input_ids', None)
                    if input_ids is not None:
                        input_ids = input_ids.to(self.device)
                        labels = labels.to(self.device)
                        if attention_mask is not None:
                            attention_mask = attention_mask.to(self.device)
                        
                        outputs = self.model(
                            eeg_signals=meg_signals,
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            labels=labels
                        )
                        generation_losses.append(outputs['lm_loss'].item())
        
        # Calculate metrics
        metrics = {}
        
        if references and all(ref for ref in references):
            # ROUGE scores
            rouge_scores = self.metrics_evaluator.compute_rouge(predictions, references)
            metrics['rouge'] = rouge_scores
            
            # BLEU scores
            bleu_scores = self.metrics_evaluator.compute_bleu(predictions, references)
            metrics['bleu'] = bleu_scores
        
        metrics['avg_generation_loss'] = float(np.mean(generation_losses)) if generation_losses else 0
        metrics['num_samples_evaluated'] = len(predictions)
        
        # Save some example generations
        examples = []
        for i in range(min(10, len(predictions))):
            examples.append({
                'prediction': predictions[i],
                'reference': references[i] if i < len(references) else None
            })
        metrics['examples'] = examples
        
        return metrics
    
    def evaluate_phase_specific_performance(self, dataloader):
        """Evaluate performance metrics for each training phase"""
        print("\n=== Evaluating Phase-Specific Components ===")
        
        results = {}
        
        with torch.no_grad():
            # Phase 1: Encoder-only metrics
            encoder_norm = 0
            num_batches = 0
            
            for batch in tqdm(dataloader, desc="Phase 1 metrics"):
                meg_signals = batch['eeg_signals'].to(self.device)
                encoded = self.model.eeg_encoder(meg_signals)
                encoder_norm += torch.mean(torch.norm(encoded, dim=-1)).item()
                num_batches += 1
                if num_batches >= 10:  # Sample a few batches
                    break
            
            results['phase1_encoder'] = {
                'avg_encoding_norm': encoder_norm / num_batches
            }
            
            # Phase 2: RVQ metrics
            vq_commitment_losses = []
            for batch in tqdm(dataloader, desc="Phase 2 metrics"):
                meg_signals = batch['eeg_signals'].to(self.device)
                encoded = self.model.eeg_encoder(meg_signals)
                _, _, aux_losses = self.model.rvq(encoded)
                if isinstance(aux_losses, dict):
                    vq_commitment_losses.append(
                        aux_losses.get('commitment_loss', torch.tensor(0)).item()
                    )
                if len(vq_commitment_losses) >= 10:
                    break
            
            results['phase2_rvq'] = {
                'avg_commitment_loss': float(np.mean(vq_commitment_losses)) if vq_commitment_losses else 0
            }
            
            # Phase 3: End-to-end metrics
            e2e_losses = []
            for batch in tqdm(dataloader, desc="Phase 3 metrics"):
                if 'labels' in batch or 'input_ids' in batch:
                    meg_signals = batch['eeg_signals'].to(self.device)
                    input_ids = batch.get('input_ids', None)
                    attention_mask = batch.get('attention_mask', None)
                    labels = batch.get('labels', input_ids)  # Use input_ids as labels if labels not present
                    
                    if input_ids is not None and labels is not None:
                        input_ids = input_ids.to(self.device)
                        labels = labels.to(self.device)
                        if attention_mask is not None:
                            attention_mask = attention_mask.to(self.device)
                        
                        outputs = self.model(
                            eeg_signals=meg_signals,
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            labels=labels
                        )
                        e2e_losses.append(outputs['loss'].item())
                if len(e2e_losses) >= 10:
                    break
            
            results['phase3_e2e'] = {
                'avg_total_loss': float(np.mean(e2e_losses)) if e2e_losses else 0
            }
        
        return results


def visualize_results(results, output_dir):
    """Create visualizations of evaluation results"""
    output_dir = Path(output_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Brain-to-Text Model Evaluation Results', fontsize=16)
    
    # 1. Codebook utilization
    ax = axes[0, 0]
    if 'encoder_reconstruction' in results:
        codebook_utils = results['encoder_reconstruction']['codebook_utilization']
        utils_data = [cb['utilization_rate'] for cb in codebook_utils.values()]
        codebook_names = [f"CB {i}" for i in range(len(utils_data))]
        ax.bar(codebook_names, utils_data)
        ax.set_xlabel('Codebook')
        ax.set_ylabel('Utilization Rate')
        ax.set_title('RVQ Codebook Utilization')
        ax.set_ylim(0, 1)
    
    # 2. Generation metrics
    ax = axes[0, 1]
    if 'generation_quality' in results and 'rouge' in results['generation_quality']:
        rouge_scores = results['generation_quality']['rouge']
        metrics = ['ROUGE-1', 'ROUGE-2', 'ROUGE-L']
        scores = [rouge_scores.get(m.lower(), 0) for m in metrics]
        ax.bar(metrics, scores)
        ax.set_xlabel('Metric')
        ax.set_ylabel('Score')
        ax.set_title('ROUGE Scores')
        ax.set_ylim(0, 1)
    
    # 3. BLEU scores
    ax = axes[1, 0]
    if 'generation_quality' in results and 'bleu' in results['generation_quality']:
        bleu_scores = results['generation_quality']['bleu']
        metrics = ['BLEU-1', 'BLEU-2', 'BLEU-3', 'BLEU-4']
        scores = [bleu_scores.get(f'bleu-{i+1}', 0) for i in range(4)]
        ax.bar(metrics, scores)
        ax.set_xlabel('Metric')
        ax.set_ylabel('Score')
        ax.set_title('BLEU Scores')
        ax.set_ylim(0, 1)
    
    # 4. Loss comparison
    ax = axes[1, 1]
    losses = []
    loss_names = []
    if 'encoder_reconstruction' in results:
        losses.append(results['encoder_reconstruction']['avg_reconstruction_error'])
        loss_names.append('Recon Error')
    if 'generation_quality' in results:
        losses.append(results['generation_quality']['avg_generation_loss'])
        loss_names.append('Gen Loss')
    if 'phase_specific' in results:
        losses.append(results['phase_specific']['phase3_e2e']['avg_total_loss'])
        loss_names.append('E2E Loss')
    
    if losses:
        ax.bar(loss_names, losses)
        ax.set_xlabel('Loss Type')
        ax.set_ylabel('Value')
        ax.set_title('Loss Comparison')
    
    plt.tight_layout()
    
    # Save figure
    fig_path = output_dir / f'evaluation_results_{timestamp}.png'
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return fig_path


def main():
    parser = argparse.ArgumentParser(description='Evaluate Brain-to-Text model')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to configuration file')
    parser.add_argument('--checkpoint', type=str, default='./output/final_model',
                       help='Path to model checkpoint')
    parser.add_argument('--output_dir', type=str, default='./evaluation_results',
                       help='Directory to save evaluation results')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use')
    parser.add_argument('--max_samples', type=int, default=50,
                       help='Maximum samples for generation evaluation')
    parser.add_argument('--encoder_only', action='store_true',
                       help='Only evaluate encoder reconstruction')
    parser.add_argument('--generation_only', action='store_true',
                       help='Only evaluate text generation')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load configuration and setup dataset
    print("Loading configuration and dataset...")
    config = load_config(args.config)
    train_dataloader, val_dataloader, tokenizer = setup_dataset(config)
    
    # Load model
    print(f"Loading model from {args.checkpoint}...")
    model = BrainToTextModel.from_pretrained(args.checkpoint)
    model.to(args.device)
    
    # Create evaluator
    evaluator = ModelEvaluator(model, tokenizer, args.device)
    
    # Run evaluations
    results = {}
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if not args.generation_only:
        # Evaluate encoder reconstruction
        encoder_results = evaluator.evaluate_encoder_reconstruction(val_dataloader)
        results['encoder_reconstruction'] = encoder_results
        print(f"\nEncoder Reconstruction Error: {encoder_results['avg_reconstruction_error']:.4f}")
        print(f"Average VQ Loss: {encoder_results['avg_vq_loss']:.4f}")
    
    if not args.encoder_only:
        # Evaluate generation quality
        generation_results = evaluator.evaluate_generation_quality(
            val_dataloader, 
            max_samples=args.max_samples
        )
        results['generation_quality'] = generation_results
        
        if 'rouge' in generation_results:
            print(f"\nROUGE Scores:")
            for metric, score in generation_results['rouge'].items():
                print(f"  {metric}: {score:.4f}")
        
        if 'bleu' in generation_results:
            print(f"\nBLEU Scores:")
            for metric, score in generation_results['bleu'].items():
                print(f"  {metric}: {score:.4f}")
    
    # Evaluate phase-specific performance
    phase_results = evaluator.evaluate_phase_specific_performance(val_dataloader)
    results['phase_specific'] = phase_results
    
    # Save results
    results_file = output_dir / f'evaluation_results_{timestamp}.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_file}")
    
    # Create visualizations
    fig_path = visualize_results(results, output_dir)
    print(f"Visualizations saved to {fig_path}")
    
    # Generate evaluation report
    report_file = output_dir / f'evaluation_report_{timestamp}.md'
    with open(report_file, 'w') as f:
        f.write("# Brain-to-Text Model Evaluation Report\n\n")
        f.write(f"**Model Checkpoint**: {args.checkpoint}\n")
        f.write(f"**Evaluation Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("## Encoder Performance\n\n")
        if 'encoder_reconstruction' in results:
            f.write(f"- Average Reconstruction Error: {results['encoder_reconstruction']['avg_reconstruction_error']:.4f}\n")
            f.write(f"- Average VQ Loss: {results['encoder_reconstruction']['avg_vq_loss']:.4f}\n\n")
            
            f.write("### Codebook Utilization\n\n")
            for cb_name, cb_stats in results['encoder_reconstruction']['codebook_utilization'].items():
                f.write(f"- {cb_name}: {cb_stats['utilized_codes']}/{cb_stats['total_codes']} "
                       f"({cb_stats['utilization_rate']*100:.1f}%)\n")
        
        f.write("\n## Generation Quality\n\n")
        if 'generation_quality' in results:
            gen_results = results['generation_quality']
            f.write(f"- Samples Evaluated: {gen_results['num_samples_evaluated']}\n")
            f.write(f"- Average Generation Loss: {gen_results['avg_generation_loss']:.4f}\n\n")
            
            if 'rouge' in gen_results:
                f.write("### ROUGE Scores\n\n")
                for metric, score in gen_results['rouge'].items():
                    f.write(f"- {metric}: {score:.4f}\n")
            
            if 'bleu' in gen_results:
                f.write("\n### BLEU Scores\n\n")
                for metric, score in gen_results['bleu'].items():
                    f.write(f"- {metric}: {score:.4f}\n")
            
            f.write("\n### Example Generations\n\n")
            for i, example in enumerate(gen_results.get('examples', [])[:5]):
                f.write(f"**Example {i+1}**:\n")
                f.write(f"- Prediction: {example['prediction']}\n")
                if example.get('reference'):
                    f.write(f"- Reference: {example['reference']}\n")
                f.write("\n")
    
    print(f"Evaluation report saved to {report_file}")
    print("\nEvaluation complete!")


if __name__ == "__main__":
    main()