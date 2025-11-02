"""
Configuration file for Legal Judgment Prediction system.
"""
import os
from typing import List, Optional


class Config:
    """Configuration class for the LJP system."""
    
    # OpenAI API Configuration
    OPENAI_API_KEY: Optional[str] = os.getenv("OPENAI_API_KEY")
    OPENAI_BASE_URL: Optional[str] = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    
    # Proxy Configuration
    HTTP_PROXY: Optional[str] = os.getenv("HTTP_PROXY")
    HTTPS_PROXY: Optional[str] = os.getenv("HTTPS_PROXY")
    
    # Default Model Settings
    DEFAULT_MODEL: str = "gpt-3.5-turbo"
    DEFAULT_DATASET: str = "cail18"
    DEFAULT_SMALL_MODEL: str = "CNN"
    DEFAULT_TASK: str = "charge"
    DEFAULT_SHOT: int = 3
    DEFAULT_BATCH_SIZE: int = 1
    DEFAULT_RETRIEVER: str = "dense_retrieval"
    
    # Text Processing
    MAX_FACT_LENGTH: int = 512
    MAX_EXEMPLAR_LENGTH: int = 256
    MAX_ARTICLE_LENGTH: int = 100
    
    # Token Limits
    MAX_TOKENS_DAVINCI: int = 4096
    MAX_TOKENS_TURBO: int = 4096
    TOKEN_BUFFER: int = 500
    
    # Task-specific prompts
    TASK_PROMPTS = {
        "charge": "本案的被告人罪名是",
        "article": "本案的相关法条是",
        "penalty": "本案的被告人刑期是",
    }
    
    # Penalty Classification
    PENALTY_CLASSES = [
        "其他", "六个月以下", "六到九个月", "九个月到一年",
        "一到两年", "二到三年", "三到五年", "五到七年",
        "七到十年", "十年以上"
    ]
    
    # Paths
    DATA_DIR: str = "data"
    OUTPUT_DIR: str = "data/output"
    
    @classmethod
    def setup_proxy(cls):
        """Setup proxy environment variables if configured."""
        if cls.HTTP_PROXY:
            os.environ["http_proxy"] = cls.HTTP_PROXY
        if cls.HTTPS_PROXY:
            os.environ["https_proxy"] = cls.HTTPS_PROXY


