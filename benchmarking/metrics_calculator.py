import re
import string
from typing import List, Dict, Tuple
from collections import Counter
import numpy as np

# Try to import optional dependencies
try:
    from bert_score import score as bert_score_func
    BERT_SCORE_AVAILABLE = True
except ImportError:
    BERT_SCORE_AVAILABLE = False
    print("⚠️  BERTScore not available. Install with: pip install bert-score")

try:
    from rouge_score import rouge_scorer
    ROUGE_AVAILABLE = True
except ImportError:
    ROUGE_AVAILABLE = False
    print("⚠️  ROUGE not available. Install with: pip install rouge-score")

try:
    import nltk
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        print("Downloading NLTK punkt tokenizer...")
        nltk.download('punkt', quiet=True)
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False


class MetricsCalculator:
    
    def __init__(self, bert_model: str = "microsoft/deberta-xlarge-mnli"):
       
        self.bert_model = bert_model
        
        # Initialize ROUGE scorer if available
        if ROUGE_AVAILABLE:
            self.rouge_scorer = rouge_scorer.RougeScorer(
                ['rouge1', 'rouge2', 'rougeL'],
                use_stemmer=True
            )
    
    def normalize_answer(self, s: str) -> str:
        
        def remove_articles(text):
            return re.sub(r'\b(a|an|the)\b', ' ', text)
        
        def white_space_fix(text):
            return ' '.join(text.split())
        
        def remove_punc(text):
            exclude = set(string.punctuation)
            return ''.join(ch for ch in text if ch not in exclude)
        
        def lower(text):
            return text.lower()
        
        return white_space_fix(remove_articles(remove_punc(lower(s))))
    
    def exact_match(self, prediction: str, ground_truth: str) -> float:
        return float(self.normalize_answer(prediction) == self.normalize_answer(ground_truth))
    
    def f1_score(self, prediction: str, ground_truth: str) -> float:
        pred_tokens = self.normalize_answer(prediction).split()
        truth_tokens = self.normalize_answer(ground_truth).split()
        
        if len(pred_tokens) == 0 or len(truth_tokens) == 0:
            return float(pred_tokens == truth_tokens)
        
        common = Counter(pred_tokens) & Counter(truth_tokens)
        num_same = sum(common.values())
        
        if num_same == 0:
            return 0.0
        
        precision = num_same / len(pred_tokens)
        recall = num_same / len(truth_tokens)
        f1 = (2 * precision * recall) / (precision + recall)
        
        return f1
    
    def precision_recall(self, prediction: str, ground_truth: str) -> Tuple[float, float]:
        pred_tokens = self.normalize_answer(prediction).split()
        truth_tokens = self.normalize_answer(ground_truth).split()
        
        if len(pred_tokens) == 0:
            return 0.0, 0.0
        if len(truth_tokens) == 0:
            return 0.0, 0.0
        
        common = Counter(pred_tokens) & Counter(truth_tokens)
        num_same = sum(common.values())
        
        precision = num_same / len(pred_tokens) if len(pred_tokens) > 0 else 0.0
        recall = num_same / len(truth_tokens) if len(truth_tokens) > 0 else 0.0
        
        return precision, recall
    
    def rouge_scores(self, prediction: str, ground_truth: str) -> Dict[str, float]:
        if not ROUGE_AVAILABLE:
            return {'rouge1': 0.0, 'rouge2': 0.0, 'rougeL': 0.0}
        
        scores = self.rouge_scorer.score(ground_truth, prediction)
        
        return {
            'rouge1': scores['rouge1'].fmeasure,
            'rouge2': scores['rouge2'].fmeasure,
            'rougeL': scores['rougeL'].fmeasure
        }
    
    def bert_score(
        self,
        predictions: List[str],
        ground_truths: List[str],
        verbose: bool = False
    ) -> Dict[str, float]:
    
        if not BERT_SCORE_AVAILABLE:
            print("⚠️  BERTScore not available. Returning zeros.")
            return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
        
        if len(predictions) == 0 or len(ground_truths) == 0:
            return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
        
        try:
            # Calculate BERTScore
            P, R, F1 = bert_score_func(
                predictions,
                ground_truths,
                model_type=self.bert_model,
                verbose=verbose,
                device='cpu'  # Use CPU for compatibility
            )
            
            return {
                'precision': float(P.mean()),
                'recall': float(R.mean()),
                'f1': float(F1.mean())
            }
        except Exception as e:
            print(f"⚠️  BERTScore calculation failed: {e}")
            return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
    
    def calculate_all_metrics(
        self,
        prediction: str,
        ground_truth: str
    ) -> Dict[str, float]:
        
        metrics = {}
        
        # Exact Match
        metrics['exact_match'] = self.exact_match(prediction, ground_truth)
        
        # F1 Score
        metrics['f1'] = self.f1_score(prediction, ground_truth)
        
        # Precision and Recall
        precision, recall = self.precision_recall(prediction, ground_truth)
        metrics['precision'] = precision
        metrics['recall'] = recall
        
        # ROUGE scores
        if ROUGE_AVAILABLE:
            rouge_scores = self.rouge_scores(prediction, ground_truth)
            metrics.update(rouge_scores)
        
        # BERTScore (single item - less efficient, use batch for multiple)
        if BERT_SCORE_AVAILABLE:
            bert_scores = self.bert_score([prediction], [ground_truth], verbose=False)
            metrics['bert_precision'] = bert_scores['precision']
            metrics['bert_recall'] = bert_scores['recall']
            metrics['bert_f1'] = bert_scores['f1']
        
        return metrics
    
    def calculate_batch_metrics(
        self,
        predictions: List[str],
        ground_truths: List[str],
        verbose: bool = False
    ) -> Dict[str, float]:
       
        if len(predictions) != len(ground_truths):
            raise ValueError("Predictions and ground truths must have same length")
        
        if len(predictions) == 0:
            return {}
        
        # Calculate per-sample metrics
        exact_matches = []
        f1_scores = []
        precisions = []
        recalls = []
        rouge1_scores = []
        rouge2_scores = []
        rougeL_scores = []
        
        for pred, truth in zip(predictions, ground_truths):
            exact_matches.append(self.exact_match(pred, truth))
            f1_scores.append(self.f1_score(pred, truth))
            
            precision, recall = self.precision_recall(pred, truth)
            precisions.append(precision)
            recalls.append(recall)
            
            if ROUGE_AVAILABLE:
                rouge = self.rouge_scores(pred, truth)
                rouge1_scores.append(rouge['rouge1'])
                rouge2_scores.append(rouge['rouge2'])
                rougeL_scores.append(rouge['rougeL'])
        
        # Aggregate results
        metrics = {
            'exact_match': np.mean(exact_matches),
            'f1': np.mean(f1_scores),
            'precision': np.mean(precisions),
            'recall': np.mean(recalls)
        }
        
        if ROUGE_AVAILABLE:
            metrics['rouge1'] = np.mean(rouge1_scores)
            metrics['rouge2'] = np.mean(rouge2_scores)
            metrics['rougeL'] = np.mean(rougeL_scores)
        
        # BERTScore (batch calculation - more efficient)
        if BERT_SCORE_AVAILABLE:
            if verbose:
                print(f"\nCalculating BERTScore for {len(predictions)} predictions...")
            bert_scores = self.bert_score(predictions, ground_truths, verbose=verbose)
            metrics['bert_precision'] = bert_scores['precision']
            metrics['bert_recall'] = bert_scores['recall']
            metrics['bert_f1'] = bert_scores['f1']
        
        return metrics


# Convenience functions for quick usage
def calculate_metrics(prediction: str, ground_truth: str) -> Dict[str, float]:
    calc = MetricsCalculator()
    return calc.calculate_all_metrics(prediction, ground_truth)


def calculate_batch_metrics(predictions: List[str], ground_truths: List[str]) -> Dict[str, float]:
    calc = MetricsCalculator()
    return calc.calculate_batch_metrics(predictions, ground_truths)


# Example usage and testing
if __name__ == "__main__":
    print("="*80)
    print("TESTING METRICS CALCULATOR")
    print("="*80)
    
    # Test data
    prediction = "The capital of France is Paris, a beautiful city."
    ground_truth = "Paris"
    
    print(f"\nPrediction: {prediction}")
    print(f"Ground Truth: {ground_truth}")
    
    # Calculate metrics
    calc = MetricsCalculator()
    metrics = calc.calculate_all_metrics(prediction, ground_truth)
    
    print("\n" + "="*80)
    print("METRICS RESULTS")
    print("="*80)
    
    for metric, value in metrics.items():
        print(f"{metric:20s}: {value:.4f}")
    
    # Test batch calculation
    print("\n" + "="*80)
    print("BATCH CALCULATION TEST")
    print("="*80)
    
    predictions = [
        "Paris is the capital of France",
        "The answer is 42",
        "I don't know"
    ]
    ground_truths = [
        "Paris",
        "42",
        "Unknown"
    ]
    
    batch_metrics = calc.calculate_batch_metrics(predictions, ground_truths, verbose=True)
    
    print("\nBatch Averaged Metrics:")
    for metric, value in batch_metrics.items():
        print(f"{metric:20s}: {value:.4f}")