"""
Evaluation script for penalty prediction task.
"""

import argparse
from typing import List

from sklearn.metrics import precision_recall_fscore_support

from src.metrics.evaluator import BaseEvaluator
from config import Config

# Penalty classification
PENALTY_CLASSES = Config.PENALTY_CLASSES


def get_penalty_class(penalty_months: int) -> int:
    """
    Convert penalty in months to classification index.
    
    Args:
        penalty_months: Penalty in months
        
    Returns:
        Classification index (0-9)
    """
    if penalty_months > 10 * 12:
        return 9
    elif penalty_months > 7 * 12:
        return 8
    elif penalty_months > 5 * 12:
        return 7
    elif penalty_months > 3 * 12:
        return 6
    elif penalty_months > 2 * 12:
        return 5
    elif penalty_months > 1 * 12:
        return 4
    elif penalty_months > 9:
        return 3
    elif penalty_months > 6:
        return 2
    elif penalty_months > 0:
        return 1
    else:
        return 0


class PenaltyEvaluator(BaseEvaluator):
    """Evaluator for penalty prediction task."""
    
    def extract_labels(self) -> List[str]:
        """Extract ground truth penalty classes."""
        labels = []
        for case in self.data:
            penalty_months = case["meta"]["term_of_imprisonment"]["imprisonment"]
            pt_cls = get_penalty_class(penalty_months)
            labels.append(PENALTY_CLASSES[pt_cls])
        return labels
    
    def extract_predictions(self) -> List[str]:
        """Extract predicted penalty classes from results."""
        predictions = []
        
        for result in self.results:
            # Support both completion and chat formats
            if "message" in result["choices"][0]:
                content = result["choices"][0]["message"]["content"]
            elif "text" in result["choices"][0]:
                content = result["choices"][0]["text"]
            else:
                content = ""
            
            # Find matching penalty class in content
            predicted_penalty = ""
            for penalty_str in PENALTY_CLASSES:
                if penalty_str in content:
                    predicted_penalty = penalty_str
                    break
            
            predictions.append(predicted_penalty)
        
        return predictions


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description="Evaluate penalty predictions")
    parser.add_argument(
        "--testset_path",
        type=str,
        required=True,
        help="Path to test set JSON file"
    )
    parser.add_argument(
        "--result_path",
        type=str,
        required=True,
        help="Path to prediction results JSON file"
    )
    
    args = parser.parse_args()
    
    evaluator = PenaltyEvaluator(args.testset_path, args.result_path)
    evaluator.print_metrics()


if __name__ == "__main__":
    main()
