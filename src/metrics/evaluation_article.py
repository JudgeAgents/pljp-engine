"""
Evaluation script for article prediction task.
"""

import re
import argparse
from typing import List

import cn2an

from src.metrics.evaluator import BaseEvaluator


class ArticleEvaluator(BaseEvaluator):
    """Evaluator for article prediction task."""
    
    def extract_labels(self) -> List[int]:
        """Extract ground truth article IDs."""
        labels = []
        for case in self.data:
            article_id = int(max(case["meta"]["relevant_articles"]))
            labels.append(article_id)
        return labels
    
    def extract_predictions(self) -> List[int]:
        """Extract predicted article IDs from results."""
        predictions = []
        
        for result in self.results:
            # Support both completion and chat formats
            if "message" in result["choices"][0]:
                content = result["choices"][0]["message"]["content"]
            elif "text" in result["choices"][0]:
                content = result["choices"][0]["text"]
            else:
                content = ""
            
            # Extract article numbers
            article_numbers = re.findall(r"第(.*?)条", content)
            article_ids = [0]  # Default
            
            for article_str in article_numbers:
                if article_str.isdigit():
                    article_ids.append(int(article_str))
                else:
                    try:
                        # Convert Chinese numbers to Arabic
                        article_id = cn2an.cn2an(article_str, mode="smart")
                        article_ids.append(int(article_id))
                    except (ValueError, TypeError):
                        article_ids.append(0)
            
            predictions.append(max(article_ids))
        
        return predictions


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description="Evaluate article predictions")
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
    
    evaluator = ArticleEvaluator(args.testset_path, args.result_path)
    evaluator.print_metrics()


if __name__ == "__main__":
    main()
