"""
Utilities module for Legal Judgment Prediction system.
"""
from .loader import (
    truncate_text,
    load_precedent,
    load_topk_option,
    load_retrieved_articles,
    load_law_articles,
)
from .prompt_gen import (
    retrieved_label_option_fewshot,
    label_prompt_case,
    fact_split,
)

__all__ = [
    "truncate_text",
    "load_precedent",
    "load_topk_option",
    "load_retrieved_articles",
    "load_law_articles",
    "retrieved_label_option_fewshot",
    "label_prompt_case",
    "fact_split",
]


