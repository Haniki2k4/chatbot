#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Setup script - Initialize project structure
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import (
    PROJECT_ROOT, DATA_DIR, RAW_DATA_DIR, CACHE_DIR, MODELS_DIR, LOGS_DIR
)


def setup():
    """Setup project structure"""
    print("Setting up project structure...\n")
    
    directories = [
        DATA_DIR,
        RAW_DATA_DIR,
        CACHE_DIR,
        MODELS_DIR,
        LOGS_DIR,
    ]
    
    for directory in directories:
        if not directory.exists():
            directory.mkdir(parents=True, exist_ok=True)
            print(f"Created: {directory.relative_to(PROJECT_ROOT)}")
        else:
            print(f"Exists: {directory.relative_to(PROJECT_ROOT)}")
    
    print("\nProject structure created successfully.")
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Data directory: {DATA_DIR}")
    print(f"Models directory: {MODELS_DIR}")
    print(f"Logs directory: {LOGS_DIR}")


if __name__ == "__main__":
    setup()
