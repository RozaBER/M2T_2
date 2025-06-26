# Encoder Evaluation Metrics Implementation Plan

## Overview
This document outlines the implementation plan for adding ROUGE and BLEU evaluation metrics to the EEG encoder module in the Large-Brain-to-Text-Model project. These metrics will help assess the quality of text generation from brain signals.

## Current Architecture Analysis

### Existing Components
- **EEGEncoder** (`src/models/eeg_encoder.py`): CNN + Transformer architecture for encoding brain signals
- **BrainToTextModel** (`src/models/full_model.py`): Complete pipeline from EEG → Encoder → RVQ → LLM → Text
- **Trainer** (`src/training/trainer.py`): Basic training loop with loss-based evaluation only

### Gap Analysis
- No text generation quality metrics (ROUGE, BLEU)
- No dedicated evaluation module for encoder performance
- Limited evaluation during training (only loss tracking)

## Implementation Outline

### 1. Create Evaluation Metrics Module
**File**: `src/evaluation/metrics.py`

```python
class EncoderEvaluationMetrics:
    """Evaluation metrics for brain-to-text encoder"""
    
    def __init__(self):
        - Initialize ROUGE scorer
        - Initialize BLEU scorer
        - Setup other metrics (METEOR, BERTScore optional)
    
    def compute_rouge(self, predictions, references):
        - Calculate ROUGE-1, ROUGE-2, ROUGE-L scores
        - Return dictionary of scores
    
    def compute_bleu(self, predictions, references):
        - Calculate BLEU-1 through BLEU-4 scores
        - Handle multiple references per prediction
        - Return dictionary of scores
    
    def compute_all_metrics(self, predictions, references):
        - Compute all available metrics
        - Return comprehensive metrics dictionary
```

### 2. Extend Encoder Module with Evaluation
**File**: `src/models/eeg_encoder.py` (modifications)

```python
class EEGEncoder(nn.Module):
    # Existing implementation...
    
    def evaluate_generation_quality(self, 
                                   eeg_inputs, 
                                   reference_texts, 
                                   model, 
                                   tokenizer,
                                   metrics_evaluator):
        """
        Evaluate encoder output quality using text generation metrics
        
        Args:
            eeg_inputs: Brain signal inputs
            reference_texts: Ground truth texts
            model: Full brain-to-text model
            tokenizer: Text tokenizer
            metrics_evaluator: EncoderEvaluationMetrics instance
        
        Returns:
            Dictionary of evaluation metrics
        """
        - Encode EEG inputs
        - Generate text predictions using full model
        - Compute metrics against reference texts
        - Return comprehensive evaluation results
```

### 3. Create Evaluation Pipeline
**File**: `src/evaluation/encoder_evaluator.py`

```python
class EncoderEvaluator:
    """Comprehensive evaluation pipeline for encoder module"""
    
    def __init__(self, encoder, full_model, tokenizer):
        - Initialize components
        - Setup metrics evaluator
        - Configure evaluation settings
    
    def evaluate_on_dataset(self, dataloader, device):
        - Process entire evaluation dataset
        - Aggregate metrics across batches
        - Generate evaluation report
    
    def evaluate_encoder_representations(self, dataloader):
        - Analyze encoder output quality
        - Compute representation metrics
        - Assess encoding consistency
```

### 4. Integration with Training Pipeline
**File**: `src/training/trainer.py` (modifications)

```python
class Trainer:
    # Existing implementation...
    
    def evaluate_with_generation_metrics(self, val_dataloader):
        """Extended evaluation with text generation metrics"""
        - Run standard evaluation (loss computation)
        - Generate texts for validation samples
        - Compute ROUGE/BLEU scores
        - Log comprehensive metrics
        - Save best model based on generation quality
```

### 5. Configuration Updates
**File**: `configs/default_config.yaml` (additions)

```yaml
evaluation:
  compute_generation_metrics: true
  metrics:
    - rouge  # ROUGE-1, ROUGE-2, ROUGE-L
    - bleu   # BLEU-1 through BLEU-4
    - meteor # Optional
    - bertscore # Optional
  
  generation_params:
    max_length: 128
    num_beams: 4
    temperature: 0.7
    top_p: 0.9
  
  evaluation_frequency: 5  # Evaluate every N epochs
  save_best_by_metric: "rouge-l"  # Metric for model selection
```

### 6. Utility Functions
**File**: `src/evaluation/utils.py`

```python
def preprocess_for_metrics(predictions, references):
    """Preprocess texts for metric computation"""
    - Tokenization normalization
    - Handle special tokens
    - Format for metric libraries

def aggregate_metrics(metric_results):
    """Aggregate metrics across batches"""
    - Compute mean, std, confidence intervals
    - Generate summary statistics

def generate_evaluation_report(metrics, save_path):
    """Generate comprehensive evaluation report"""
    - Create detailed metrics breakdown
    - Generate visualization plots
    - Save results to file
```

## Implementation Steps

### Phase 1: Core Infrastructure (Week 1)
1. Create `src/evaluation/` directory structure
2. Implement `EncoderEvaluationMetrics` class with ROUGE and BLEU
3. Add required dependencies to `requirements.txt`:
   ```
   rouge-score>=0.1.2
   sacrebleu>=2.3.1
   nltk>=3.8
   ```

### Phase 2: Encoder Integration (Week 2)
1. Extend `EEGEncoder` with evaluation methods
2. Create `EncoderEvaluator` class
3. Implement evaluation pipeline with proper error handling

### Phase 3: Training Integration (Week 3)
1. Modify `Trainer` class to support generation metrics
2. Update training loop to periodically evaluate with new metrics
3. Implement metric-based model checkpointing

### Phase 4: Testing and Optimization (Week 4)
1. Create comprehensive test suite for evaluation metrics
2. Optimize evaluation speed (batching, caching)
3. Add visualization tools for metric analysis

## Usage Example

```python
from src.models.eeg_encoder import EEGEncoder
from src.models.full_model import BrainToTextModel
from src.evaluation.metrics import EncoderEvaluationMetrics
from src.evaluation.encoder_evaluator import EncoderEvaluator

# Initialize components
encoder = EEGEncoder(config)
model = BrainToTextModel(config)
tokenizer = AutoTokenizer.from_pretrained(config.llm_model)
evaluator = EncoderEvaluator(encoder, model, tokenizer)

# Evaluate on validation set
results = evaluator.evaluate_on_dataset(val_dataloader, device)
print(f"ROUGE-L: {results['rouge-l']:.4f}")
print(f"BLEU-4: {results['bleu-4']:.4f}")

# During training
trainer = Trainer(model, config)
trainer.evaluate_with_generation_metrics(val_dataloader)
```

## Expected Benefits

1. **Quantitative Assessment**: Objective measurement of text generation quality
2. **Model Selection**: Choose best checkpoints based on generation metrics
3. **Research Insights**: Understand relationship between encoder representations and text quality
4. **Debugging Tool**: Identify encoding issues affecting generation quality

## Future Extensions

1. **Additional Metrics**:
   - METEOR for semantic similarity
   - BERTScore for contextual embeddings
   - Custom brain-to-text specific metrics

2. **Encoder Analysis**:
   - Attention visualization
   - Representation clustering analysis
   - Temporal encoding quality assessment

3. **Online Evaluation**:
   - Real-time metric computation during training
   - Early stopping based on generation quality
   - Adaptive learning rate based on metrics

## Dependencies

```bash
pip install rouge-score sacrebleu nltk torch-metrics
```

## Conclusion

This implementation plan provides a comprehensive approach to adding ROUGE and BLEU evaluation metrics to the encoder module. The modular design allows for easy extension with additional metrics and seamless integration with the existing training pipeline.