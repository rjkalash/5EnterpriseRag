"""
Configuration settings for the Enterprise RAG Knowledge Base.
"""
from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # OpenAI Configuration
    openai_api_key: str
    openai_model: str = "gpt-4-turbo-preview"
    embedding_model: str = "text-embedding-3-small"
    
    # Qdrant Configuration
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_api_key: Optional[str] = None
    qdrant_collection_name: str = "enterprise_knowledge_base"
    
    # Application Settings
    app_host: str = "0.0.0.0"
    app_port: int = 8000
    log_level: str = "INFO"
    environment: str = "development"
    
    # Hybrid Search Configuration
    sparse_weight: float = 0.3
    dense_weight: float = 0.7
    top_k_results: int = 5
    
    # RAG Configuration
    chunk_size: int = 1000
    chunk_overlap: int = 200
    max_context_length: int = 4000
    temperature: float = 0.7
    
    # Ragas Evaluation
    enable_evaluation: bool = True
    evaluation_sample_size: int = 10
    
    class Config:
        env_file = ".env"
        case_sensitive = False


# Global settings instance
settings = Settings()
