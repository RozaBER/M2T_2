import numpy as np
from typing import List, Dict, Union, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os


def preprocess_for_metrics(predictions: List[str], references: List[str]) -> Tuple[List[str], List[str]]:
    """
    Preprocess texts for metric computation
    
    Args:
        predictions: List of predicted texts
        references: List of reference texts
        
    Returns:
        Preprocessed predictions and references
    """
    # Remove extra whitespace and normalize
    predictions = [' '.join(pred.strip().split()) for pred in predictions]
    references = [' '.join(ref.strip().split()) for ref in references]
    
    # Handle empty predictions
    predictions = [pred if pred else "[EMPTY]" for pred in predictions]
    
    return predictions, references


def aggregate_metrics(metric_results: List[Dict[str, float]]) -> Dict[str, Union[float, Dict]]:
    """
    Aggregate metrics across batches
    
    Args:
        metric_results: List of metric dictionaries from each batch
        
    Returns:
        Aggregated metrics with mean, std, and confidence intervals
    """
    if not metric_results:
        return {}
    
    # Get all metric names
    all_metrics = set()
    for result in metric_results:
        all_metrics.update(result.keys())
    
    aggregated = {}
    
    for metric in all_metrics:
        values = []
        for result in metric_results:
            if metric in result and isinstance(result[metric], (int, float)):
                values.append(result[metric])
        
        if values:
            aggregated[metric] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values),
                'count': len(values)
            }
            
            # Add 95% confidence interval
            if len(values) > 1:
                sem = np.std(values) / np.sqrt(len(values))
                ci = 1.96 * sem
                aggregated[metric]['ci_lower'] = aggregated[metric]['mean'] - ci
                aggregated[metric]['ci_upper'] = aggregated[metric]['mean'] + ci
    
    # Also compute simple mean values for easy access
    aggregated['mean_values'] = {
        metric: aggregated[metric]['mean'] 
        for metric in aggregated 
        if isinstance(aggregated[metric], dict) and 'mean' in aggregated[metric]
    }
    
    return aggregated


def generate_evaluation_report(metrics: Dict, save_path: str):
    """
    Generate comprehensive evaluation report with visualizations
    
    Args:
        metrics: Dictionary of evaluation metrics
        save_path: Path to save the report
    """
    # Create report content
    report_lines = [
        "# Encoder Evaluation Report",
        f"\nGenerated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "\n## Summary Metrics\n"
    ]
    
    # Add corpus-level metrics if available
    if 'corpus_level' in metrics:
        report_lines.append("### Corpus-Level Metrics")
        for metric, value in metrics['corpus_level'].items():
            if isinstance(value, float):
                report_lines.append(f"- **{metric.upper()}**: {value:.4f}")
        report_lines.append("")
    
    # Add mean values
    if 'mean_values' in metrics:
        report_lines.append("### Average Metrics Across Batches")
        for metric, value in metrics['mean_values'].items():
            report_lines.append(f"- **{metric}**: {value:.4f}")
        report_lines.append("")
    
    # Add dataset statistics
    if 'dataset_stats' in metrics:
        report_lines.append("### Dataset Statistics")
        for stat, value in metrics['dataset_stats'].items():
            if isinstance(value, (int, float)):
                report_lines.append(f"- **{stat.replace('_', ' ').title()}**: {value:.2f}")
        report_lines.append("")
    
    # Add detailed metrics
    report_lines.append("## Detailed Metrics\n")
    
    for metric_name, metric_data in metrics.items():
        if isinstance(metric_data, dict) and 'mean' in metric_data:
            report_lines.append(f"### {metric_name}")
            report_lines.append(f"- Mean: {metric_data['mean']:.4f}")
            report_lines.append(f"- Std: {metric_data['std']:.4f}")
            report_lines.append(f"- Min: {metric_data['min']:.4f}")
            report_lines.append(f"- Max: {metric_data['max']:.4f}")
            if 'ci_lower' in metric_data:
                report_lines.append(f"- 95% CI: [{metric_data['ci_lower']:.4f}, {metric_data['ci_upper']:.4f}]")
            report_lines.append("")
    
    # Create visualizations
    viz_dir = os.path.dirname(save_path)
    viz_path = create_metric_visualizations(metrics, viz_dir)
    if viz_path:
        report_lines.append(f"\n## Visualizations\n")
        report_lines.append(f"![Metrics Visualization]({os.path.basename(viz_path)})")
    
    # Write report
    with open(save_path, 'w') as f:
        f.write('\n'.join(report_lines))
    
    print(f"Evaluation report saved to: {save_path}")


def create_metric_visualizations(metrics: Dict, output_dir: str) -> Optional[str]:
    """
    Create visualizations for metrics
    
    Args:
        metrics: Dictionary of metrics
        output_dir: Directory to save visualizations
        
    Returns:
        Path to saved visualization or None
    """
    try:
        # Extract plottable metrics
        plot_data = {}
        
        if 'corpus_level' in metrics:
            for metric, value in metrics['corpus_level'].items():
                if isinstance(value, (int, float)) and not metric.endswith('std'):
                    plot_data[metric] = value
        
        if not plot_data:
            return None
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Bar plot of metrics
        metrics_names = list(plot_data.keys())
        metrics_values = list(plot_data.values())
        
        bars = ax1.bar(metrics_names, metrics_values)
        ax1.set_xlabel('Metrics')
        ax1.set_ylabel('Score')
        ax1.set_title('Evaluation Metrics')
        ax1.set_ylim(0, 1.1)
        
        # Add value labels on bars
        for bar, value in zip(bars, metrics_values):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{value:.3f}', ha='center', va='bottom')
        
        # Rotate x labels
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # Create grouped bar plot for ROUGE and BLEU scores
        rouge_scores = {k: v for k, v in plot_data.items() if 'rouge' in k}
        bleu_scores = {k: v for k, v in plot_data.items() if 'bleu' in k}
        
        if rouge_scores and bleu_scores:
            x = np.arange(len(rouge_scores))
            width = 0.35
            
            rouge_values = list(rouge_scores.values())
            bleu_values = list(bleu_scores.values())[:len(rouge_values)]
            
            ax2.bar(x - width/2, rouge_values, width, label='ROUGE', color='skyblue')
            ax2.bar(x + width/2, bleu_values[:len(x)], width, label='BLEU', color='lightcoral')
            
            ax2.set_xlabel('N-gram Level')
            ax2.set_ylabel('Score')
            ax2.set_title('ROUGE vs BLEU Scores')
            ax2.set_xticks(x)
            ax2.set_xticklabels(['1-gram', '2-gram', '3-gram', '4-gram'][:len(x)])
            ax2.legend()
            ax2.set_ylim(0, 1.1)
        
        plt.tight_layout()
        
        # Save figure
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        viz_path = os.path.join(output_dir, f'metrics_visualization_{timestamp}.png')
        plt.savefig(viz_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return viz_path
        
    except Exception as e:
        print(f"Error creating visualizations: {e}")
        return None