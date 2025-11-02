"""
Data loading utilities for Legal Judgment Prediction system.
"""

import json
import os
import re
from typing import List, Dict, Any, Optional

import cn2an
from tqdm import tqdm

from config import Config


def truncate_text(text: str, max_len: int = 1024) -> str:
    """
    Truncate text by keeping prefix and suffix within length limit.
    
    Args:
        text: Input text to truncate
        max_len: Maximum length allowed
        
    Returns:
        Truncated text
    """
    if not text:
        return ""
    
    # Add spaces after punctuation for splitting
    text = text.replace("。", "。 ").replace("，", "， ").replace("；", "； ").replace("、", "、 ")
    text = text.split()
    
    # Remove "经审理" prefix if present
    if text and "经审理" in text[0]:
        text = text[1:]
    
    if not text:
        return ""
    
    # Keep prefix and suffix
    prefix = []
    postfix = []
    n = len(text)
    i, j = 0, n - 1
    
    # Build prefix up to half of max_len
    while i < n:
        sent = text[i]
        prefix.append(sent)
        if len(" ".join(prefix)) >= max_len // 2:
            break
        i += 1
    
    # Build postfix up to half of max_len
    while j > i:
        sent = text[j]
        postfix.insert(0, sent)
        if len(" ".join(postfix)) >= max_len // 2:
            break
        j -= 1
    
    ret_text = prefix + postfix
    return "".join(ret_text)


def load_precedent(
    pool_path: str, 
    precedent_idx_path: str
) -> List[List[Dict[str, Any]]]:
    """
    Load precedents based on indices.
    
    Args:
        pool_path: Path to precedent pool JSON file
        precedent_idx_path: Path to precedent indices JSON file
        
    Returns:
        List of precedent lists for each case
    """
    # Load precedent pool
    pool = []
    with open(pool_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                obj = json.loads(line)
                pool.append(obj)
    
    # Load precedent indices
    with open(precedent_idx_path, "r", encoding="utf-8") as f:
        precedent_idxs = json.load(f)
    
    # Retrieve precedents for each case
    precedents = []
    for precedent_idx in tqdm(precedent_idxs, desc="Loading similar cases"):
        similar_cases = [pool[idx] for idx in precedent_idx]
        precedents.append(similar_cases)
    
    return precedents


def ar_idx2text(article_id: int, index_type: str = "str") -> str:
    """
    Convert article index to text format.
    
    Args:
        article_id: Article ID number
        index_type: "str" for Chinese text, "num" for number
        
    Returns:
        Formatted article text (e.g., "第二百六十三条")
    """
    if index_type == "str":
        astr = cn2an.an2cn(article_id)
    elif index_type == "num":
        astr = str(article_id)
    else:
        raise ValueError(f"Unknown index_type: {index_type}")
    
    return f"第{astr}条"


def load_law_articles(path: Optional[str] = None) -> Dict[str, str]:
    """
    Load law articles from file.
    
    Args:
        path: Path to law articles file. Defaults to Config.DATA_DIR/output/meta/laws.txt
        
    Returns:
        Dictionary mapping article text to article content
    """
    if path is None:
        path = os.path.join(Config.DATA_DIR, "output", "meta", "laws.txt")
    
    laws = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            laws.append(line.strip())
    
    aid = 0
    article_map = {}
    
    for line in laws:
        astr = ar_idx2text(aid)
        arr = line.split()
        
        # Check if next article starts
        if len(arr) and arr[0] == ar_idx2text(aid + 1):
            astr = ar_idx2text(aid + 1, index_type="str")
            article_map[astr] = line
            aid += 1
        else:
            # Continue current article
            if astr in article_map:
                article_map[astr] += line
            else:
                article_map[astr] = line
    
    # Clean up: remove article header from content
    article_map = {
        k: v.replace(k, "").strip() 
        for k, v in article_map.items()
    }
    
    return article_map


def load_topk_option(path: str) -> List[List[str]]:
    """
    Load top-k label options from file.
    
    Args:
        path: Path to top-k options JSON file
        
    Returns:
        List of top-k label lists for each case
    """
    with open(path, "r", encoding="utf-8") as f:
        topk_label_option = json.load(f)
    return topk_label_option


def load_retrieved_articles(
    path: str, 
    index_type: str = "str"
) -> List[List[str]]:
    """
    Load retrieved articles with their definitions.
    
    Args:
        path: Path to retrieved article indices JSON file
        index_type: "str" for Chinese text, "num" for number
        
    Returns:
        List of article definition lists for each case
    """
    # Load article indices
    with open(path, "r", encoding="utf-8") as f:
        retrieved_ar_lst_idx = json.load(f)
    
    # Load law articles mapping
    law_map = load_law_articles()
    
    # Convert indices to article definitions
    retrieved_ar_lst = []
    for lst in tqdm(retrieved_ar_lst_idx, desc="Loading article content"):
        cur_ar_lst = []
        for aid in lst:
            astr = ar_idx2text(aid, index_type="str")
            anum = ar_idx2text(aid, index_type="num")
            article_content = law_map.get(astr, "")
            cur_ar_lst.append(f"{anum}：{article_content}")
        retrieved_ar_lst.append(cur_ar_lst)
    
    return retrieved_ar_lst


def load_step1_resp(path: str) -> List[str]:
    """
    Load step 1 responses from output file.
    
    Args:
        path: Path to output JSON file
        
    Returns:
        List of response texts
    """
    responses = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                obj = json.loads(line)
                # Support both completion and chat formats
                if "text" in obj["choices"][0]:
                    text = obj["choices"][0]["text"]
                elif "message" in obj["choices"][0]:
                    text = obj["choices"][0]["message"]["content"]
                else:
                    text = ""
                responses.append(text)
    return responses


def load_predicted_article_content(predicted_article: str) -> str:
    """
    Load content for a predicted article.
    
    Args:
        predicted_article: Article string (may contain article number)
        
    Returns:
        Article content text
    """
    law_map = load_law_articles()
    
    # Extract article number
    numbers = re.findall(r"\d+", predicted_article)
    if numbers:
        article_num = int(numbers[0])
    else:
        return ""
    
    # Validate article number
    if article_num < 0 or article_num > 451:
        return ""
    
    # Get article content
    astr = ar_idx2text(article_num, index_type="str")
    article_content = law_map.get(astr, "")
    
    return article_content
