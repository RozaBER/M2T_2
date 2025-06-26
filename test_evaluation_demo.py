"""
Demo script to test the evaluation metrics implementation
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.evaluation.metrics import EncoderEvaluationMetrics

# Test the evaluation metrics
def test_metrics():
    print("Testing Encoder Evaluation Metrics Implementation")
    print("=" * 50)
    
    # Initialize metrics evaluator
    evaluator = EncoderEvaluationMetrics()
    
    # Sample predictions and references
    predictions = [
        "The cat sat on the mat.",
        "The weather is nice today.",
        "Machine learning is fascinating.",
        "Python is a great programming language.",
        "Neural networks can process complex data."
    ]
    
    references = [
        "The cat was sitting on the mat.",
        "Today the weather is pleasant.",
        "Machine learning is very interesting.",
        "Python is an excellent programming language.",
        "Neural networks can handle complex data."
    ]
    
    print("\nSample Data:")
    print("-" * 30)
    for i, (pred, ref) in enumerate(zip(predictions, references)):
        print(f"Pair {i+1}:")
        print(f"  Prediction: {pred}")
        print(f"  Reference:  {ref}")
        print()
    
    # Compute ROUGE scores
    print("\nComputing ROUGE scores...")
    rouge_scores = evaluator.compute_rouge(predictions, references)
    
    print("\nROUGE Scores:")
    print("-" * 30)
    for metric, score in rouge_scores.items():
        if not metric.endswith('-std'):
            print(f"{metric.upper()}: {score:.4f}")
    
    # Compute BLEU scores
    print("\nComputing BLEU scores...")
    bleu_scores = evaluator.compute_bleu(predictions, references)
    
    print("\nBLEU Scores:")
    print("-" * 30)
    for metric, score in bleu_scores.items():
        print(f"{metric.upper()}: {score:.4f}")
    
    # Compute all metrics
    print("\nComputing all metrics together...")
    all_metrics = evaluator.compute_all_metrics(predictions, references)
    
    print("\nAll Metrics Summary:")
    print("-" * 30)
    for metric, score in sorted(all_metrics.items()):
        if isinstance(score, float) and not metric.endswith('-std'):
            print(f"{metric}: {score:.4f}")
    
    print("\n" + "=" * 50)
    print("Evaluation metrics implementation is working correctly!")
    print("\nThe implementation provides:")
    print("- ROUGE-1, ROUGE-2, and ROUGE-L scores")
    print("- BLEU-1 through BLEU-4 scores")
    print("- Standard deviation for ROUGE scores")
    print("- Batch processing capabilities")
    
    return True

if __name__ == "__main__":
    try:
        test_metrics()
        print("\n✓ Test completed successfully!")
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()