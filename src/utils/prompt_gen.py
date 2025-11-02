"""
Prompt generation utilities for Legal Judgment Prediction system.
"""

import json
import os
import re
from argparse import Namespace
from typing import Dict, Any, List

from config import Config
from src.utils import loader


# Penalty classification strings
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


def label_prompt_case(case: Dict[str, Any], task: str) -> str:
    """
    Extract label from case for prompt generation.
    
    Args:
        case: Case dictionary
        task: Task type ("charge", "article", or "penalty")
        
    Returns:
        Formatted label string
    """
    if task == "charge":
        label = case["meta"]["accusation"][0]
    elif task == "article":
        article_id = max(case["meta"]["relevant_articles"])
        label = f"第{article_id}条"
    elif task == "penalty":
        penalty_months = case["meta"]["term_of_imprisonment"]["imprisonment"]
        pt_cls = get_penalty_class(penalty_months)
        label = PENALTY_CLASSES[pt_cls] + "有期徒刑"
    else:
        raise ValueError(f"Unknown task: {task}")
    
    return label


def format_label_for_prompt(label: str, task: str) -> str:
    """
    Format label for prompt inclusion.
    
    Args:
        label: Raw label string
        task: Task type
        
    Returns:
        Formatted label string
    """
    if task == "article":
        # Ensure article format
        if not label.startswith("第"):
            label = f"第{label}条"
    return label


def retrieved_label_option_fewshot(
    fact: str,
    in_context_content: Dict[str, Any],
    args: Namespace,
    case_id: str
) -> str:
    """
    Generate few-shot prompt with retrieved labels and precedents.
    
    Args:
        fact: Case fact description
        in_context_content: Dictionary containing precedents, top-k labels, etc.
        args: Command line arguments
        case_id: Case ID
        
    Returns:
        Generated prompt string
    """
    # Load predicted articles and charges (for dependent tasks)
    predicted_articles = {}
    predicted_charges = {}
    
    if args.predicted_article_path and os.path.exists(args.predicted_article_path):
        with open(args.predicted_article_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    case = json.loads(line)
                    case_id_key = case.get("caseID", "")
                    # Extract article number from response
                    content = case["choices"][0].get("message", {}).get("content", "")
                    numbers = re.findall(r'\d+', content)
                    if numbers:
                        predicted_articles[case_id_key] = numbers[0]
    
    if args.predicted_charge_path and os.path.exists(args.predicted_charge_path):
        with open(args.predicted_charge_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    case = json.loads(line)
                    case_id_key = case.get("caseID", "")
                    content = case["choices"][0].get("message", {}).get("content", "")
                    # Get first line of charge prediction
                    predicted_charges[case_id_key] = content.split('\n\n')[0].strip()
    
    # Get predicted article and charge for current case
    predicted_article = predicted_articles.get(case_id, "")
    predicted_article_content = ""
    if predicted_article:
        article_content = loader.load_predicted_article_content(predicted_article)
        predicted_article_content = loader.truncate_text(
            article_content, 
            max_len=Config.MAX_ARTICLE_LENGTH
        )
    
    predicted_charge = predicted_charges.get(case_id, "")
    
    # Task-specific prompt prefix
    task_prompt = Config.TASK_PROMPTS[args.task]
    
    # Prepare few-shot examples from precedents
    precedents = in_context_content.get("precedents", [])
    label_options = in_context_content.get("topk_label_option", [])
    
    # Build example texts
    example_texts = []
    for precedent_case in precedents[:args.shot]:
        cur_fact = precedent_case["fact"].replace(" ", "")
        
        if args.use_split_fact:
            fact_parts = [
                precedent_case["fact_split"]["zhuguan"],
                precedent_case["fact_split"]["keguan"],
                precedent_case["fact_split"]["shiwai"]
            ]
            cur_fact = "。".join(fact_parts)
        
        cur_label = label_prompt_case(precedent_case, args.task)
        cur_fact = loader.truncate_text(cur_fact, max_len=Config.MAX_EXEMPLAR_LENGTH)
        
        example_text = f"{cur_fact}\n{task_prompt}：{cur_label}"
        example_texts.append(example_text)
    
    # Format label options
    formatted_labels = [
        format_label_for_prompt(label, args.task) 
        for label in label_options[:args.shot]
    ]
    label_options_str = "；".join(formatted_labels)
    
    # Add article definitions for article task
    if args.task == "article":
        article_definitions = in_context_content.get("article_definition", [])
        article_definitions = [
            loader.truncate_text(text, max_len=50) 
            for text in article_definitions[:args.shot]
        ]
        example_texts.extend(article_definitions)
    
    # Combine examples
    examples_text = "\n\n".join(example_texts)
    if examples_text:
        examples_text += "\n\n"
    
    # Generate task-specific prompt
    if args.task == "article":
        prompt = (
            f"用<>括起来的是本案事实，用---括起来的是与本案内容类似的{args.shot}个案件。"
            f"请通过主观动机，客观行为与事外情节三个方面，理解与比较用---括起来的{args.shot}个案件与本案事实的异同，"
            f"选择本案的相关法条。注意：请输出本案的相关法条，并结合类案给出选择本案相关法条的理由。\n"
            f"<{fact}>\n"
            f"---{examples_text}---\n"
            f"{task_prompt}以下几个选项的其中之一：[{label_options_str}]\n"
            f"{task_prompt}："
        )
    elif args.task == "charge":
        prompt = (
            f"用<>括起来的是本案事实，用---括起来的是与本案内容类似的{args.shot}个案件，"
            f"用```括起来的是本案的相关法条和法条内容。"
            f"请通过主观动机，客观行为与事外情节三个方面，理解与比较用---括起来的{args.shot}个案件与本案事实的异同，"
            f"并参考本案的相关法条和法条内容，输出本案的罪名。"
            f"注意：请输出本案的罪名，并结合类案给出选择本案罪名的理由。\n"
            f"<{fact}>\n"
            f"---{examples_text}---\n"
            f"```{predicted_article} {predicted_article_content}```\n"
            f"{task_prompt}以下几个选项的其中之一：[{label_options_str}]\n"
            f"{task_prompt}："
        )
    else:  # penalty
        prompt = (
            f"用<>括起来的是本案事实，用---括起来的是与本案内容类似的{args.shot}个案件，"
            f"用```括起来的是本案的相关法条、法条内容和相关罪名。"
            f"请通过主观动机，客观行为与事外情节三个方面，理解与比较用---括起来的{args.shot}个案件与本案事实的异同，"
            f"并参考本案的相关法条、法条内容和相关罪名，选择最终的刑期。"
            f"注意：请输出本案的刑期，并结合类案给出选择本案刑期的理由。\n"
            f"<{fact}>\n"
            f"---{examples_text}---\n"
            f"```{predicted_article} {predicted_article_content} {predicted_charge}```\n"
            f"{task_prompt}以下几个选项的其中之一：[{label_options_str}]\n"
            f"{task_prompt}："
        )
    
    return prompt


def fact_split(fact: str) -> str:
    """
    Generate prompt for fact splitting task.
    
    Args:
        fact: Raw fact description
        
    Returns:
        Prompt for LLM to split fact into components
    """
    prompt = f"""
一段法院查明的犯罪事实可以区分成：主观动机、客观行为以及事外情节。
其中，主观动机是指行为人对自己实施的危害社会的行为及其结果所持的心理态度，包括犯罪的故意、过失，犯罪的动机、目的等。
客观行为是指构成犯罪在客观活动方面所必须具备的条件，包括危害行为、危害结果，以及危害行为与危害结果之间的因果关系等。
事外情节是指决定刑罚轻重时根据的各种事实情况，从轻处罚的情节包括自首、有立功表现等，从重处罚的情节包括累犯等。
下面提供两个参考示例。根据以上信息，你的任务是将用```括起来的犯罪事实进行归纳，并总结成与参考示例同样的格式。
字数限制在200字以内。

示例:
事实：经审理查明，被告人徐佑华舞厅老板，携带事先在农药店买好的农药，来到舞厅，将农药随机投放在舞厅进门处座位上的两杯茶水里。经鉴定，两杯茶水中均检出农药毒死蜱成分。被告人徐佑华被公安机关抓获归案。但被告人徐佑华归案后如实供述其罪行，依法可以从轻处罚。
主观动机：被告人徐佑华在公共场所故意投放毒害性物质
客观行为：被告人徐佑华舞厅老板，携带事先在农药店买好的农药，来到舞厅，将农药随机投放在舞厅进门处座位上的两杯茶水里。经鉴定，两杯茶水中均检出农药毒死蜱成分。
事外情节：但被告人徐佑华归案后如实供述其罪行，依法可以从轻处罚。

```{fact}```
"""
    return prompt
