"""
Evaluation script for charge prediction task.
"""

import argparse
from collections import Counter
from typing import List, Set, Dict, Tuple

from sklearn.metrics import precision_recall_fscore_support

from src.metrics.evaluator import BaseEvaluator


# Charge matching patterns for fuzzy matching
LAIC_CHARGE_MATCH = {
    '非法生产、买卖、运输制毒物品、走私制毒物品': [
        ('生产', '制毒物品'), ('买卖', '制毒物品'), 
        ('运输', '制毒物品'), ('走私', '制毒物品')
    ],
    '非法经营': [('非法', '经营')],
    '非法转让、倒卖土地使用权': [
        ('转让', '土地'), ('倒卖', '土地')
    ]
}

CAIL_CHARGE_MATCH = {
    '重大劳动安全事故': [('劳动', '事故')],
    '容留他人吸毒': [('容留', '吸毒')],
    '非法种植毒品原植物': [
        ('种', '毒品'), ('植', '毒品')
    ],
    '盗伐林木': [
        ('盗', '伐', '林'), ('盗', '伐', '木'),
        ('偷', '伐', '林'), ('偷', '伐', '木')
    ],
    '故意杀人': [('故意', '杀人')],
    '交通肇事': [('肇事',)],
    '污染环境': [('污染',)],
    '强奸': [('强奸',)],
    '合同诈骗': [('合同', '诈骗')],
    '生产、销售不符合安全标准的食品': [
        ('产', '不', '安全', '食'), ('售', '不', '安全', '食'),
        ('卖', '不', '安全', '食')
    ],
    '强制猥亵、侮辱妇女': [
        ('猥亵', '女'), ('侮辱', '女')
    ],
    '妨害信用卡管理': [('信用卡', '管理')],
    '赌博': [('赌博',)],
    '生产、销售伪劣产品': [
        ('产', '伪', '品'), ('产', '劣', '品'),
        ('售', '伪', '品'), ('售', '劣', '品'),
        ('卖', '伪', '品'), ('卖', '劣', '品')
    ],
    '妨害公务': [('妨', '公务')],
    '职务侵占': [
        ('职务', '侵'), ('职务', '占')
    ],
    '非法采矿': [('采矿',)],
    '滥用职权': [
        ('滥用', '职'), ('滥用', '权')
    ],
    '破坏广播电视设施、公用电信设施': [
        ('破坏', '广播'), ('破坏', '电信')
    ],
    '放火': [('放火',)],
    '伪造、变造、买卖国家机关公文、证件、印章': [
        ('伪造', '印章'), ('伪造', '公章')
    ],
    '非法采伐、毁坏国家重点保护植物': [
        ('采', '保护', '植'), ('伐', '保护', '植'),
        ('毁', '保护', '植'), ('坏', '保护', '植')
    ],
    '开设赌场': [
        ('开', '赌场'), ('开', '设')
    ],
    '生产、销售假药': [
        ('产', '假', '药'), ('售', '假', '药'),
        ('卖', '假', '药')
    ],
    '非法吸收公众存款': [
        ('吸', '公众', '款'), ('收', '公众', '款')
    ],
    '玩忽职守': [('忽', '职守')],
}

TOTAL_CHARGE_MATCH = {**LAIC_CHARGE_MATCH, **CAIL_CHARGE_MATCH}


def get_similar_charge(text: str, charge_patterns: Dict[str, List[Tuple]]) -> str:
    """
    Find similar charge using pattern matching.
    
    Args:
        text: Prediction text
        charge_patterns: Dictionary of charge -> patterns
        
    Returns:
        Matched charge or "#" if not found
    """
    matched_charges = set()
    
    for charge, patterns in charge_patterns.items():
        for pattern_words in patterns:
            if all(word in text for word in pattern_words):
                matched_charges.add(charge)
    
    if not matched_charges:
        return "#"
    
    # Return longest match (most specific)
    return sorted(matched_charges, key=len, reverse=True)[0]


class ChargeEvaluator(BaseEvaluator):
    """Evaluator for charge prediction task."""
    
    def __init__(self, testset_path: str, result_path: str):
        """Initialize charge evaluator."""
        super().__init__(testset_path, result_path)
        self.charge_set = self._get_charge_set()
    
    def _get_charge_set(self) -> Set[str]:
        """Get set of all possible charges from test set."""
        charges = [case["meta"]["accusation"][0] for case in self.data]
        return set(charges)
    
    def extract_labels(self) -> List[str]:
        """Extract ground truth charges."""
        return [case["meta"]["accusation"][0] for case in self.data]
    
    def extract_predictions(self) -> List[str]:
        """Extract predicted charges from results."""
        predictions = []
        
        for result in self.results:
            # Support both completion and chat formats
            if "message" in result["choices"][0]:
                content = result["choices"][0]["message"]["content"]
            elif "text" in result["choices"][0]:
                content = result["choices"][0]["text"]
            else:
                content = ""
            
            # First try exact match
            matched_charges = set()
            for charge in self.charge_set:
                if charge in content:
                    matched_charges.add(charge)
            
            if len(matched_charges) == 1:
                predictions.append(list(matched_charges)[0])
            else:
                # Try fuzzy matching
                predicted_charge = get_similar_charge(content, TOTAL_CHARGE_MATCH)
                predictions.append(predicted_charge)
        
        return predictions


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description="Evaluate charge predictions")
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
    
    evaluator = ChargeEvaluator(args.testset_path, args.result_path)
    evaluator.print_metrics()


if __name__ == "__main__":
    main()
