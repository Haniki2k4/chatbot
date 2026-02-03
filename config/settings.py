# -*- coding: utf-8 -*-
"""
Cấu hình toàn cục cho Chatbot
"""
import os
from pathlib import Path

# ==================== Đường dẫn ====================
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
CACHE_DIR = DATA_DIR / "cache"
MODELS_DIR = DATA_DIR / "models"
LOGS_DIR = PROJECT_ROOT / "logs"

# Tạo các thư mục nếu chưa tồn tại
for directory in [DATA_DIR, RAW_DATA_DIR, CACHE_DIR, MODELS_DIR, LOGS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# ==================== Chatbot Configuration ====================
CHATBOT_CONFIG = {
    "data_folder": str(RAW_DATA_DIR / "duc_giang_txt"),
    "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
    "chunk_size": 3,
    "top_k": 5,
    "similarity_threshold": 0.3,
    "cache_file": str(CACHE_DIR / "chatbot_cache.pkl"),
    "use_llm": True,
    "max_contexts": 3,
    "llm_model_path": str(MODELS_DIR / "qwen2.5-1.5b-instruct-q4_k_m.gguf"),
}

# ==================== LLM Configuration ====================
LLM_CONFIG = {
    "n_ctx": 1024,  # Context window
    "n_threads": max(1, os.cpu_count() // 2),
    "n_batch": 64,
    "verbose": False,
    "temperature": 0.3,
    "top_p": 0.9,
    "max_tokens": 400,
}

# ==================== Flask Configuration ====================
FLASK_CONFIG = {
    "host": "0.0.0.0",
    "port": 5000,
    "debug": False,
    "threaded": True,
}

# ==================== Logging Configuration ====================
LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "standard": {
            "format": "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
        },
        "detailed": {
            "format": "%(asctime)s [%(levelname)s] %(name)s (%(filename)s:%(lineno)d): %(message)s"
        },
    },
    "handlers": {
        "default": {
            "level": "INFO",
            "class": "logging.StreamHandler",
            "formatter": "standard",
        },
        "file": {
            "level": "DEBUG",
            "class": "logging.handlers.RotatingFileHandler",
            "filename": str(LOGS_DIR / "chatbot.log"),
            "maxBytes": 1024 * 1024,  # 1MB
            "backupCount": 3,
            "formatter": "detailed",
        },
    },
    "loggers": {
        "": {
            "handlers": ["default", "file"],
            "level": "INFO",
            "propagate": True,
        }
    },
}

# ==================== Preprocess Version ====================
PREPROCESS_VERSION = "v0.1.12"
