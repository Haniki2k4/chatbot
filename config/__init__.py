"""
Configuration module for chatbot application
"""
import os
from pathlib import Path

# Get project root
BASE_DIR = Path(__file__).parent.parent

# Default config
class Config:
    """Base configuration"""
    
    # Flask
    FLASK_APP = "app.py"
    FLASK_ENV = "development"
    DEBUG = False
    
    # Paths
    BASE_DIR = BASE_DIR
    DATA_DIR = BASE_DIR / "data"
    DOCUMENTS_DIR = DATA_DIR / "documents"
    CACHE_DIR = DATA_DIR / "cache"
    MODELS_DIR = BASE_DIR / "models"
    
    # Embeddings
    EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    CHUNK_SIZE = 3
    PREPROCESS_VERSION = "v4"
    
    # LLM
    USE_LLM = True
    LLM_MODEL_PATH = MODELS_DIR / "qwen2.5-1.5b-instruct-q4_k_m.gguf"
    LLM_N_CTX = 512
    LLM_N_THREADS = max(1, os.cpu_count() // 2)
    LLM_N_BATCH = 64
    
    # Cache
    CACHE_FILE = CACHE_DIR / "chatbot_cache.pkl"
    
    # Server
    HOST = "0.0.0.0"
    PORT = 5000
    WORKERS = 1
    
    # Thresholds
    SIMILARITY_THRESHOLD = 0.3
    TOP_K = 5

class DevelopmentConfig(Config):
    """Development configuration"""
    DEBUG = True
    FLASK_ENV = "development"

class ProductionConfig(Config):
    """Production configuration for Hugging Face Spaces"""
    DEBUG = False
    FLASK_ENV = "production"
    WORKERS = 4
    HOST = "0.0.0.0"
    PORT = int(os.getenv("PORT", 7860))  # HF Spaces default port

class TestingConfig(Config):
    """Testing configuration"""
    TESTING = True
    CACHE_FILE = CACHE_DIR / "test_cache.pkl"

# Config selector
config_dict = {
    "development": DevelopmentConfig,
    "production": ProductionConfig,
    "testing": TestingConfig,
    "default": DevelopmentConfig
}

def get_config(env=None):
    """Get config based on environment"""
    if env is None:
        env = os.getenv("FLASK_ENV", "development")
    return config_dict.get(env, config_dict["default"])
