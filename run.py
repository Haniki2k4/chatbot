#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Startup script - Run chatbot web server
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

if __name__ == "__main__":
    from web.app import app, init_chatbot, FLASK_CONFIG
    
    # Pre-initialize chatbot
    print("Initializing chatbot...")
    try:
        init_chatbot()
        print("Chatbot ready.")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
    
    # Start Flask server
    app.run(
        host=FLASK_CONFIG['host'],
        port=FLASK_CONFIG['port'],
        debug=FLASK_CONFIG['debug'],
        threaded=FLASK_CONFIG['threaded']
    )
