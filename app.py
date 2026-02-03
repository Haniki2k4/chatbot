# -*- coding: utf-8 -*-
"""
Entry point for Hugging Face Spaces deployment
Compatible with web/app.py module
"""
import sys
import os
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

# Load environment
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Import Flask app from web module
from web.app import app, init_chatbot

if __name__ == '__main__':
    # Initialize chatbot before running
    init_chatbot()
    
    # Run Flask
    port = int(os.getenv('PORT', 7860))
    debug = os.getenv('FLASK_DEBUG', 'False') == 'True'
    
    print("\n" + "="*60)
    print("🏥 CHATBOT BỆnh VIỆN ĐỨC GIANG")
    print("="*60)
    print(f"📍 URL: http://localhost:{port}")
    print(f"🔧 Debug: {debug}")
    print("="*60 + "\n")
    
    app.run(
        host='0.0.0.0',
        port=port,
        debug=debug,
        threaded=True
    )
