# -*- coding: utf-8 -*-
"""
Flask web server for Duc Giang Hospital Chatbot
"""

import os
import logging
from flask import Flask, render_template, request, jsonify
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import config and chatbot
from config import get_config
from src.chatbot_engine import DucGiangChatbot

# Initialize Flask app
app = Flask(__name__, 
            template_folder=str(Path(__file__).parent / "templates"),
            static_folder=str(Path(__file__).parent / "static"))

# Get config
config = get_config()
app.config.from_object(config)

# Global chatbot instance
chatbot = None


def init_chatbot():
    """Initialize chatbot (singleton pattern)"""
    global chatbot
    if chatbot is None:
        logger.info("Initializing chatbot...")
        chatbot = DucGiangChatbot(config=config)
    return chatbot


@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')


@app.route('/api/chat', methods=['POST'])
def chat():
    """
    Chat API endpoint
    
    Request:
        {
            "message": "user question",
            "top_k": 5 (optional)
        }
    
    Response:
        {
            "response": "answer",
            "time": total_time,
            "inference_time": llm_inference_time,
            "scores": [confidence scores]
        }
    """
    try:
        data = request.get_json()
        message = data.get('message', '').strip()
        top_k = data.get('top_k', config.TOP_K)
        
        if not message:
            return jsonify({'error': 'Please enter a question'}), 400
        
        # Get response from chatbot
        bot = init_chatbot()
        response, scores, inference_time = bot.get_response(
            message,
            top_k=top_k,
            return_scores=True
        )
        
        return jsonify({
            'response': response,
            'inference_time': inference_time,
            'scores': scores[:2]
        })
    
    except Exception as e:
        logger.error(f"Chat API error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/stats', methods=['GET'])
def stats():
    """Get chatbot statistics"""
    try:
        bot = init_chatbot()
        stats_data = bot.get_stats()
        return jsonify(stats_data)
    except Exception as e:
        logger.error(f"Stats API error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/health', methods=['GET'])
def health():
    """Health check"""
    return jsonify({'status': 'ok', 'message': 'Chatbot is running'})


@app.errorhandler(404)
def not_found(e):
    return jsonify({'error': 'Endpoint not found'}), 404


@app.errorhandler(500)
def server_error(e):
    return jsonify({'error': 'Server error'}), 500


if __name__ == '__main__':
    # Initialize chatbot
    init_chatbot()
    
    logger.info("\n" + "="*60)
    logger.info("🌐 CHATBOT SERVER")
    logger.info("="*60)
    logger.info(f"📍 URL: http://localhost:{config.PORT}")
    logger.info(f"🔧 Environment: {config.FLASK_ENV}")
    logger.info("="*60 + "\n")
    
    app.run(
        host=config.HOST,
        port=config.PORT,
        debug=config.DEBUG,
        threaded=True
    )
