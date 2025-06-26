"""
Simplified script to evaluate the encoder module with ROUGE and BLEU metrics
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
from src.evaluation.metrics import EncoderEvaluationMetrics
from src.evaluation.utils import generate_evaluation_report


def test_evaluation_metrics():
    """Test the evaluation metrics with dummy data"""
    print("Testing Encoder Evaluation Metrics")
    print("=" * 50)
    
    # Initialize metrics evaluator
    evaluator = EncoderEvaluationMetrics()
    
    # Create dummy data for testing
    predictions = [
        "The person is reading a book.",
        "The subject is listening to music.",
        "The participant is watching a video.",
        "The individual is thinking about food.",
        "The person is solving a math problem."
    ]
    
    references = [
        "The person was reading a book quietly.",
        "The subject was listening to classical music.",
        "The participant watched an interesting video.",
        "The individual thought about their lunch.",
        "The person solved a complex math problem."
    ]
    
    print("\nTest Data:")
    print("-" * 30)
    for i, (pred, ref) in enumerate(zip(predictions, references)):
        print(f"Sample {i+1}:")
        print(f"  Prediction: {pred}")
        print(f"  Reference:  {ref}")
    
    # Compute all metrics
    print("\nComputing metrics...")
    metrics = evaluator.compute_all_metrics(predictions, references)
    
    print("\nEvaluation Results:")
    print("-" * 30)
    for metric, score in sorted(metrics.items()):
        if isinstance(score, float) and not metric.endswith('-std'):
            print(f"{metric}: {score:.4f}")
    
    # Test encoder architecture
    print("\n" + "=" * 50)
    print("Testing Encoder Architecture")
    print("=" * 50)
    
    # Create a dummy encoder
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    encoder = EEGEncoder(
        n_channels=208,
        sampling_rate=1000,
        segment_length=0.1,
        d_model=512,
        n_heads=8,
        n_layers=6,
        d_ff=2048,
        dropout=0.1
    ).to(device)
    
    # Create dummy EEG input
    batch_size = 2
    n_channels = 208
    time_points = 1000  # 1 second of data
    dummy_eeg = torch.randn(batch_size, n_channels, time_points).to(device)
    
    print(f"\nInput shape: {dummy_eeg.shape}")
    
    # Forward pass
    with torch.no_grad():
        encoded_features = encoder(dummy_eeg)
    
    print(f"Encoded features shape: {encoded_features.shape}")
    print(f"Feature dimensionality: {encoded_features.shape[-1]}")
    
    # Generate evaluation report
    print("\n" + "=" * 50)
    print("Generating Evaluation Report")
    print("=" * 50)
    
    output_dir = "./evaluation_results"
    os.makedirs(output_dir, exist_ok=True)
    
    report_metrics = {
        'corpus_level': metrics,
        'dataset_stats': {
            'total_samples': len(predictions),
            'avg_prediction_length': sum(len(p.split()) for p in predictions) / len(predictions),
            'avg_reference_length': sum(len(r.split()) for r in references) / len(references)
        }
    }
    
    report_path = os.path.join(output_dir, "demo_report.md")
    generate_evaluation_report(report_metrics, report_path)
    
    print(f"Report saved to: {report_path}")
    
    print("\n" + "=" * 50)
    print("Evaluation System Summary:")
    print("=" * 50)
    print("✓ EncoderEvaluationMetrics class - Working")
    print("✓ ROUGE scores computation - Working")
    print("✓ BLEU scores computation - Working")
    print("✓ EEGEncoder forward pass - Working")
    print("✓ Evaluation report generation - Working")
    print("\nThe evaluation system is ready to use!")
    print("\nTo evaluate on real data:")
    print("1. Ensure MASC-MEG dataset is available")
    print("2. Train or load a pre-trained model")
    print("3. Run evaluation with the full pipeline")


if __name__ == "__main__":
    test_evaluation_metrics()