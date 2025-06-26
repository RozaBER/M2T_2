import numpy as np
from typing import List, Dict, Union, Tuple
from rouge_score import rouge_scorer
from sacrebleu import corpus_bleu
import nltk
from collections import defaultdict

class EncoderEvaluationMetrics:
    """Evaluation metrics for brain-to-text encoder"""
    
    def __init__(self):
        """Initialize evaluation metrics"""
        self.rouge_types = ['rouge1', 'rouge2', 'rougeL']
        self.rouge_scorer = rouge_scorer.RougeScorer(self.rouge_types, use_stemmer=True)
        
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
    
    def compute_rouge(self, predictions: List[str], references: List[str]) -> Dict[str, float]:
        """
        Calculate ROUGE scores for predictions against references
        
        Args:
            predictions: List of predicted texts
            references: List of reference texts
            
        Returns:
            Dictionary containing ROUGE-1, ROUGE-2, and ROUGE-L F1 scores
        """
        scores = defaultdict(list)
        
        for pred, ref in zip(predictions, references):
            rouge_scores = self.rouge_scorer.score(ref, pred)
            for rouge_type in self.rouge_types:
                scores[rouge_type].append(rouge_scores[rouge_type].fmeasure)
        
        return {
            'rouge-1': np.mean(scores['rouge1']),
            'rouge-2': np.mean(scores['rouge2']),
            'rouge-l': np.mean(scores['rougeL']),
            'rouge-1-std': np.std(scores['rouge1']),
            'rouge-2-std': np.std(scores['rouge2']),
            'rouge-l-std': np.std(scores['rougeL'])
        }
    
    def compute_bleu(self, predictions: List[str], references: List[List[str]]) -> Dict[str, float]:
        """
        Calculate BLEU scores for predictions against references
        
        Args:
            predictions: List of predicted texts
            references: List of reference texts (can be multiple references per prediction)
            
        Returns:
            Dictionary containing BLEU-1 through BLEU-4 scores
        """
        if not isinstance(references[0], list):
            references = [[ref] for ref in references]
        
        bleu_scores = {}
        
        # Compute BLEU scores for all n-grams at once
        try:
            from sacrebleu.metrics import BLEU
            bleu = BLEU()
            # sacrebleu expects references as a list of lists
            if isinstance(references[0], str):
                references = [[ref] for ref in references]
            score = bleu.corpus_score(predictions, references)
            
            # Extract individual n-gram precisions
            precisions = score.precisions
            bleu_scores['bleu-1'] = precisions[0] / 100.0 if len(precisions) > 0 else 0.0
            bleu_scores['bleu-2'] = precisions[1] / 100.0 if len(precisions) > 1 else 0.0
            bleu_scores['bleu-3'] = precisions[2] / 100.0 if len(precisions) > 2 else 0.0
            bleu_scores['bleu-4'] = score.score / 100.0  # Overall BLEU-4 score
        except Exception as e:
            print(f"Error computing BLEU scores: {e}")
            for n in range(1, 5):
                bleu_scores[f'bleu-{n}'] = 0.0
        
        return bleu_scores
    
    def compute_all_metrics(self, predictions: List[str], references: List[str]) -> Dict[str, float]:
        """
        Compute all available metrics
        
        Args:
            predictions: List of predicted texts
            references: List of reference texts
            
        Returns:
            Comprehensive dictionary of all metrics
        """
        if not predictions or not references:
            return {
                'rouge-1': 0.0, 'rouge-2': 0.0, 'rouge-l': 0.0,
                'bleu-1': 0.0, 'bleu-2': 0.0, 'bleu-3': 0.0, 'bleu-4': 0.0
            }
        
        rouge_scores = self.compute_rouge(predictions, references)
        bleu_scores = self.compute_bleu(predictions, references)
        
        return {**rouge_scores, **bleu_scores}
    
    def compute_batch_metrics(self, prediction_batches: List[List[str]], 
                            reference_batches: List[List[str]]) -> Dict[str, float]:
        """
        Compute metrics for multiple batches
        
        Args:
            prediction_batches: List of prediction batches
            reference_batches: List of reference batches
            
        Returns:
            Aggregated metrics across all batches
        """
        all_predictions = []
        all_references = []
        
        for pred_batch, ref_batch in zip(prediction_batches, reference_batches):
            all_predictions.extend(pred_batch)
            all_references.extend(ref_batch)
        
        return self.compute_all_metrics(all_predictions, all_references)