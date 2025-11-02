"""
Base evaluator class for Legal Judgment Prediction tasks.
"""

import json
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Tuple

import numpy as np
from sklearn.metrics import precision_recall_fscore_support


class BaseEvaluator(ABC):
    """Base class for evaluation metrics."""
    
    def __init__(self, testset_path: str, result_path: str):
        """
        Initialize evaluator.
        
        Args:
            testset_path: Path to test set JSON file
            result_path: Path to result JSON file
        """
        self.testset_path = testset_path
        self.result_path = result_path
        self.data = self._load_data()
        self.results = self._load_results()
        
    def _load_data(self) -> List[Dict[str, Any]]:
        """Load test set data."""
        data = []
        with open(self.testset_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
        return data
    
    def _load_results(self) -> List[Dict[str, Any]]:
        """Load prediction results."""
        results = []
        with open(self.result_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    results.append(json.loads(line))
        return results
    
    @abstractmethod
    def extract_predictions(self) -> List[Any]:
        """Extract predictions from results."""
        pass
    
    @abstractmethod
    def extract_labels(self) -> List[Any]:
        """Extract ground truth labels from data."""
        pass
    
    def evaluate(self) -> Dict[str, float]:
        """
        Run evaluation and return metrics.
        
        Returns:
            Dictionary of metric name -> value
        """
        y_true = self.extract_labels()
        y_pred = self.extract_predictions()
        
        # Calculate metrics
        micro_acc, _, _, _ = precision_recall_fscore_support(
            y_true, y_pred, average='micro', zero_division=0
        )
        macro_p, macro_r, macro_f, _ = precision_recall_fscore_support(
            y_true, y_pred, average='macro', zero_division=0
        )
        
        # Format as percentages
        metrics = {
            "accuracy": round(micro_acc * 100, 2),
            "macro_precision": round(macro_p * 100, 2),
            "macro_recall": round(macro_r * 100, 2),
            "macro_f1": round(macro_f * 100, 2),
        }
        
        return metrics
    
    def print_metrics(self) -> None:
        """Print evaluation metrics."""
        metrics = self.evaluate()
        print(f"Accuracy: {metrics['accuracy']:.2f}%")
        print(f"Macro Precision: {metrics['macro_precision']:.2f}%")
        print(f"Macro Recall: {metrics['macro_recall']:.2f}%")
        print(f"Macro F1: {metrics['macro_f1']:.2f}%")


