"""
Legal Judgment Prediction using LLM and Domain-Model Collaboration.

This module implements the main inference pipeline for predicting legal judgments
using Large Language Models (LLMs) with in-context learning enhanced by domain models
and precedent retrieval.
"""

import json
import logging
import os
import time
from argparse import ArgumentParser, Namespace
from typing import Dict, List, Optional, Callable, Any

import openai
import tiktoken
from tqdm import tqdm

from config import Config
from src.utils import loader, prompt_gen

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize configuration
Config.setup_proxy()

# API Configuration
API_KEY_POOL: List[str] = []
if Config.OPENAI_API_KEY:
    API_KEY_POOL.append(Config.OPENAI_API_KEY)


class LLMResponseHandler:
    """Handles LLM API responses for different models."""
    
    @staticmethod
    def davinci_response(text_list: List[str], model_name: str) -> Dict[str, Any]:
        """
        Generate response using Davinci models.
        
        Args:
            text_list: List of input texts
            model_name: Name of the Davinci model
            
        Returns:
            API response dictionary
        """
        enc = tiktoken.get_encoding("p50k_base")
        max_tokens = Config.MAX_TOKENS_DAVINCI - max(len(enc.encode(t)) for t in text_list)
        
        response = openai.Completion.create(
            model=model_name,
            prompt=text_list,
            max_tokens=max_tokens,
            temperature=0
        )
        return response
    
    @staticmethod
    def turbo_response(text_list: List[str]) -> Dict[str, Any]:
        """
        Generate response using GPT-3.5-turbo model.
        
        Args:
            text_list: List of input texts (must be length 1)
            
        Returns:
            API response dictionary
        """
        if len(text_list) != 1:
            raise ValueError("gpt-3.5-turbo requires batch size of 1")
        
        text = text_list[0]
        enc = tiktoken.get_encoding("cl100k_base")
        max_tokens = Config.MAX_TOKENS_TURBO - len(enc.encode(text)) - Config.TOKEN_BUFFER
        
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": text}],
            max_tokens=max_tokens,
            temperature=0
        )
        return response


class LLMInferenceEngine:
    """Main inference engine for LLM-based legal judgment prediction."""
    
    def __init__(self, args: Namespace):
        """
        Initialize the inference engine.
        
        Args:
            args: Command line arguments
        """
        self.args = args
        self.response_handlers = {
            "text-davinci-002": lambda x: LLMResponseHandler.davinci_response(x, "text-davinci-002"),
            "text-davinci-003": lambda x: LLMResponseHandler.davinci_response(x, "text-davinci-003"),
            "gpt-3.5-turbo": LLMResponseHandler.turbo_response,
        }
        self.current_key_index = 0
        
    def _setup_api_key(self) -> None:
        """Setup OpenAI API key from pool."""
        if not API_KEY_POOL:
            raise ValueError("No API keys available in pool")
        openai.api_key = API_KEY_POOL[self.current_key_index]
        logger.info(f"Using API key index {self.current_key_index}")
    
    def _handle_api_error(self, error: Exception, key_pool: List[str]) -> bool:
        """
        Handle API errors with retry logic.
        
        Args:
            error: The exception that occurred
            key_pool: Pool of API keys
            
        Returns:
            True if should retry, False otherwise
        """
        error_str = repr(error).lower()
        logger.warning(f"API Error: {error}")
        
        if isinstance(error, openai.error.RateLimitError):
            if "limit" in error_str:
                logger.info("Rate limit hit, waiting 60 seconds...")
                time.sleep(60)
                return True
            elif "quota" in error_str:
                if not key_pool:
                    logger.error("All API keys exhausted!")
                    return False
                logger.info(f"Switching to next API key: {key_pool[0]}")
                openai.api_key = key_pool.pop(0)
                self.current_key_index += 1
                time.sleep(1)
                return True
        elif isinstance(error, openai.error.APIError):
            logger.warning("API error, waiting 60 seconds before retry...")
            time.sleep(60)
            return True
        
        logger.error(f"Unexpected error: {error}")
        return False
    
    def _preprocess_cases(
        self, 
        data: List[Dict[str, Any]], 
        in_context_contents: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Preprocess cases for LLM inference.
        
        Args:
            data: List of case data
            in_context_contents: In-context learning content
            
        Returns:
            List of processed prompts with case IDs
        """
        processed_data = []
        
        for i in tqdm(range(len(data)), desc="Preprocessing prompts"):
            case = data[i]
            case_id = case["caseID"]
            
            # Extract fact description
            fact = case["fact"].replace(" ", "")
            if self.args.use_split_fact:
                fact_parts = [
                    case["fact_split"]["zhuguan"],
                    case["fact_split"]["keguan"],
                    case["fact_split"]["shiwai"]
                ]
                fact = "。".join(fact_parts)
            
            # Truncate fact to max length
            fact = loader.truncate_text(fact, max_len=Config.MAX_FACT_LENGTH)
            
            # Generate prompt
            prompt = prompt_gen.retrieved_label_option_fewshot(
                fact, in_context_contents[i], self.args, case_id
            )
            
            processed_data.append({
                "prompt": prompt,
                "caseID": case_id
            })
        
        return processed_data
    
    def _get_api_response(
        self, 
        prompt_list: List[Dict[str, Any]], 
        response_function: Callable
    ) -> List[Dict[str, Any]]:
        """
        Get API response for a batch of prompts.
        
        Args:
            prompt_list: List of prompts with case IDs
            response_function: Function to call the API
            
        Returns:
            List of API responses with case IDs
        """
        text_list = [item["prompt"] for item in prompt_list]
        
        try:
            response = response_function(text_list)
            responses = []
            
            for i, prompt_item in enumerate(prompt_list):
                response_item = response.copy()
                if "choices" in response:
                    response_item["choices"] = [response["choices"][i]]
                response_item["caseID"] = prompt_item["caseID"]
                responses.append(response_item)
            
            return responses
            
        except (openai.error.RateLimitError, openai.error.APIError) as e:
            raise e
        except Exception as e:
            logger.error(f"Unexpected error in API call: {e}")
            raise
    
    def run_inference(
        self, 
        data: List[Dict[str, Any]], 
        in_context_contents: List[Dict[str, Any]],
        key_pool: List[str]
    ) -> None:
        """
        Run inference on the dataset.
        
        Args:
            data: List of case data
            in_context_contents: In-context learning content
            key_pool: Pool of API keys
        """
        if len(data) != len(in_context_contents):
            raise ValueError("Data and in-context contents must have same length")
        
        self._setup_api_key()
        response_function = self.response_handlers[self.args.model]
        
        # Preprocess all cases
        processed_data = self._preprocess_cases(data, in_context_contents)
        
        # Load existing results to resume
        existing_results = []
        if os.path.exists(self.args.output_path):
            try:
                with open(self.args.output_path, "r", encoding="utf-8") as f:
                    existing_results = [line for line in f.readlines() if line.strip()]
            except Exception as e:
                logger.warning(f"Could not load existing results: {e}")
        
        # Skip already processed cases
        processed_data = processed_data[len(existing_results):]
        logger.info(f"Processing {len(processed_data)} cases (skipped {len(existing_results)} already processed)")
        
        # Batch processing
        batch_size = self.args.batch
        i = 0
        
        logger.info("Starting inference...")
        while i < len(processed_data):
            batch_data = processed_data[i:i + batch_size]
            logger.info(f"Processing batch {i // batch_size + 1} ({i}/{len(processed_data)})")
            
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"First prompt in batch:\n{batch_data[0]['prompt'][:200]}...")
            
            retry = True
            while retry:
                try:
                    responses = self._get_api_response(batch_data, response_function)
                    
                    # Save results
                    with open(self.args.output_path, "a", encoding="utf-8") as f:
                        for resp in responses:
                            line = json.dumps(resp, ensure_ascii=False)
                            f.write(line + "\n")
                            f.flush()
                    
                    retry = False
                    time.sleep(1)  # Rate limiting
                    
                except (openai.error.RateLimitError, openai.error.APIError) as e:
                    retry = self._handle_api_error(e, key_pool)
                    if not retry:
                        raise
                except Exception as e:
                    logger.error(f"Fatal error: {e}")
                    raise
            
            i += batch_size


def load_data(input_path: str) -> List[Dict[str, Any]]:
    """
    Load case data from JSON file.
    
    Args:
        input_path: Path to input JSON file
        
    Returns:
        List of case dictionaries
    """
    data = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                case = json.loads(line)
                data.append(case)
    return data


def setup_paths(args: Namespace) -> None:
    """
    Setup default paths based on arguments.
    
    Args:
        args: Command line arguments
    """
    # Input path
    if not args.input_path:
        prefix = f"data/testset/{args.dataset}/"
        filename = "testset_fact_split.json" if args.use_split_fact else "testset.json"
        args.input_path = os.path.join(prefix, filename)
    
    # Output path
    if not args.output_path:
        args.output_path = os.path.join(
            "data", "output", "llm_out",
            args.dataset, args.small_model, args.task,
            f"{args.shot}shot", f"{args.model}.json"
        )
        os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
        # Create empty file if doesn't exist
        if not os.path.exists(args.output_path):
            with open(args.output_path, "w", encoding="utf-8") as f:
                pass
    
    # Precedent paths
    if not args.precedent_pool_path:
        prefix = "data/precedent_database/"
        filename = "precedent_case_fact_split.json" if args.use_split_fact else "precedent_case.json"
        args.precedent_pool_path = os.path.join(prefix, filename)
    
    if not args.precedent_idx_path:
        args.precedent_idx_path = os.path.join(
            "data", "output", "domain_model_out", "precedent_idx",
            args.dataset, args.retriever, f"precedent_idxs_{args.task}.json"
        )
    
    # Top-k label option path
    if not args.topk_label_option_path:
        args.topk_label_option_path = os.path.join(
            "data", "output", "domain_model_out", "candidate_label",
            args.dataset, args.small_model, f"{args.task}_topk.json"
        )
    
    # Predicted article path
    if not args.predicted_article_path:
        args.predicted_article_path = os.path.join(
            "data", "output", "llm_out",
            args.dataset, args.small_model, "article",
            f"{args.shot}shot", f"{args.model}.json"
        )
    
    # Predicted charge path
    if not args.predicted_charge_path:
        args.predicted_charge_path = os.path.join(
            "data", "output", "llm_out",
            args.dataset, args.small_model, "charge",
            f"{args.shot}shot", f"{args.model}.json"
        )


def prepare_in_context_content(args: Namespace, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Prepare in-context learning content for all cases.
    
    Args:
        args: Command line arguments
        data: List of case data
        
    Returns:
        List of in-context content dictionaries
    """
    # Load existing results to resume
    existing_results = []
    if os.path.exists(args.output_path):
        try:
            with open(args.output_path, "r", encoding="utf-8") as f:
                existing_results = [line for line in f.readlines() if line.strip()]
        except Exception:
            pass
    
    # Load precedents
    precedents = loader.load_precedent(args.precedent_pool_path, args.precedent_idx_path)
    precedents = precedents[len(existing_results):]
    
    # Load top-k label options
    topk_label_option = loader.load_topk_option(args.topk_label_option_path)
    topk_label_option = topk_label_option[len(existing_results):]
    
    # Load article definitions (for article task)
    if args.task == "article":
        article_definition = loader.load_retrieved_articles(
            args.topk_label_option_path, index_type="num"
        )
        article_definition = article_definition[len(existing_results):]
    else:
        article_definition = [["#"] for _ in range(len(precedents))]
    
    # Combine into in-context content
    in_context_contents = [
        {
            "precedents": precedents[i],
            "topk_label_option": topk_label_option[i],
            "article_definition": article_definition[i],
        }
        for i in range(len(precedents))
    ]
    
    return in_context_contents


def main(args: Namespace) -> None:
    """Main execution function."""
    # Setup paths
    setup_paths(args)
    
    # Load data
    logger.info(f"Loading data from {args.input_path}")
    data = load_data(args.input_path)
    
    # Prepare in-context content
    logger.info("Preparing in-context learning content...")
    in_context_contents = prepare_in_context_content(args, data)
    
    # Skip already processed cases
    existing_results = []
    if os.path.exists(args.output_path):
        try:
            with open(args.output_path, "r", encoding="utf-8") as f:
                existing_results = [line for line in f.readlines() if line.strip()]
        except Exception:
            pass
    
    data = data[len(existing_results):]
    in_context_contents = in_context_contents[len(existing_results):]
    
    logger.info(f"Total cases to process: {len(data)}")
    
    # Run inference
    engine = LLMInferenceEngine(args)
    engine.run_inference(data, in_context_contents, API_KEY_POOL.copy())


if __name__ == "__main__":
    parser = ArgumentParser(
        description="Legal Judgment Prediction using LLM and Domain-Model Collaboration"
    )
    
    # Model and dataset settings
    parser.add_argument(
        "--model", 
        type=str, 
        default=Config.DEFAULT_MODEL,
        choices=["text-davinci-002", "text-davinci-003", "gpt-3.5-turbo"],
        help="LLM model to use"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=Config.DEFAULT_DATASET,
        choices=["cail18", "cjo22"],
        help="Dataset to use"
    )
    parser.add_argument(
        "--small_model",
        type=str,
        default=Config.DEFAULT_SMALL_MODEL,
        choices=["CNN", "TopJudge", "ELE"],
        help="Domain model to use"
    )
    parser.add_argument(
        "--task",
        type=str,
        default=Config.DEFAULT_TASK,
        choices=["charge", "article", "penalty"],
        help="Task type"
    )
    parser.add_argument(
        "--shot",
        type=int,
        default=Config.DEFAULT_SHOT,
        help="Number of few-shot examples"
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=Config.DEFAULT_BATCH_SIZE,
        help="Batch size (for throughput optimization)"
    )
    parser.add_argument(
        "--retriever",
        type=str,
        default=Config.DEFAULT_RETRIEVER,
        choices=["bm25", "dense_retrieval"],
        help="Precedent retrieval method"
    )
    parser.add_argument(
        "--use_split_fact",
        action="store_true",
        help="Use split fact (subjective/objective/post-facto) instead of raw fact"
    )
    
    # Path arguments
    parser.add_argument("--input_path", type=str, default="", help="Test set path")
    parser.add_argument("--output_path", type=str, default="", help="Output path for LLM responses")
    parser.add_argument("--precedent_pool_path", type=str, default="", help="Precedent database path")
    parser.add_argument("--precedent_idx_path", type=str, default="", help="Precedent index path")
    parser.add_argument("--topk_label_option_path", type=str, default="", help="Top-k label options path")
    parser.add_argument("--predicted_article_path", type=str, default="", help="Predicted articles path (for charge/penalty tasks)")
    parser.add_argument("--predicted_charge_path", type=str, default="", help="Predicted charges path (for penalty task)")
    
    args = parser.parse_args()
    main(args)
