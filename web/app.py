# -*- coding: utf-8 -*-
"""
Flask Web Server cho Chatbot Bệnh viện Đức Giang
API endpoints cho frontend giao tiếp với chatbot
"""

import logging
import logging.config
from pathlib import Path
import time

from flask import Flask, render_template, request, jsonify

from config.settings import FLASK_CONFIG, LOGGING_CONFIG, CHATBOT_CONFIG
from src.chatbot_engine import DucGiangChatbot

# Cấu hình logging
logging.config.dictConfig(LOGGING_CONFIG)
logger = logging.getLogger(__name__)

# Khởi tạo Flask app
app = Flask(__name__, 
            template_folder=str(Path(__file__).parent / "templates"),
            static_folder=str(Path(__file__).parent / "static"))

# Biến global chatbot
chatbot = None


def init_chatbot():
    """Khởi tạo chatbot (chỉ khởi tạo 1 lần)"""
    global chatbot
    if chatbot is None:
        try:
            logger.info("Đang khởi tạo chatbot...")
            chatbot = DucGiangChatbot(config=CHATBOT_CONFIG)
            logger.info("Chatbot khởi tạo thành công.")
        except Exception as e:
            logger.error(f"Lỗi khi khởi tạo chatbot: {e}")
            raise
    return chatbot


@app.route('/')
def index():
    """Trang chủ - giao diện chatbot"""
    return render_template('index.html')


@app.route('/api/chat', methods=['POST'])
def chat():
    """
    API endpoint nhận câu hỏi và trả về câu trả lời
    
    Request JSON:
        {
            "message": "câu hỏi của user",
            "top_k": 5 (optional)
        }
    
    Response JSON:
        {
            "response": "câu trả lời",
            "time": thời gian xử lý tổng (seconds),
            "inference_time": thời gian suy luận (seconds),
            "scores": [thông tin xác suất]
        }
    """
    try:
        # Lấy dữ liệu từ request
        data = request.get_json()
        user_message = data.get('message', '').strip()
        top_k = data.get('top_k', 5)
        
        if not user_message:
            return jsonify({'error': 'Vui lòng nhập câu hỏi'}), 400
        
        # Đo thời gian xử lý
        start_time = time.time()
        
        # Lấy câu trả lời từ chatbot
        bot = init_chatbot()
        response, scores, inference_time = bot.get_response(
            user_message,
            top_k=top_k,
            return_scores=True
        )
        
        end_time = time.time()
        processing_time = round(end_time - start_time, 2)
        inference_time_rounded = round(inference_time, 2)
        
        logger.info(f"Query: {user_message[:50]}... | Time: {processing_time}s | Inference: {inference_time_rounded}s")
        
        # Trả về kết quả
        return jsonify({
            'response': response,
            'time': processing_time,
            'inference_time': inference_time_rounded,
            'scores': scores[:2]  # Chỉ trả về top 2 scores
        })
    
    except Exception as e:
        logger.error(f"Lỗi: {str(e)}")
        return jsonify({'error': f'Đã xảy ra lỗi: {str(e)}'}), 500


@app.route('/api/stats', methods=['GET'])
def stats():
    """
    API endpoint lấy thống kê về chatbot
    
    Response JSON:
        {
            "total_chunks": số lượng chunks,
            "embedding_dim": chiều vector embedding,
            "model": tên model,
            "llm_enabled": có LLM không,
            "llm_model": đường dẫn model LLM
        }
    """
    try:
        bot = init_chatbot()
        stats_data = bot.get_stats()
        return jsonify(stats_data)
    except Exception as e:
        logger.error(f"Lỗi khi lấy stats: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'ok',
        'message': 'Chatbot is running'
    })


@app.errorhandler(404)
def not_found(e):
    """Xử lý 404 error"""
    return jsonify({'error': 'Không tìm thấy endpoint'}), 404


@app.errorhandler(500)
def server_error(e):
    """Xử lý 500 error"""
    return jsonify({'error': 'Lỗi server'}), 500


if __name__ == '__main__':
    # Khởi tạo chatbot trước khi start server
    init_chatbot()
    
    print("\n" + "="*60)
    print("SERVER CHATBOT BỆNH VIỆN ĐỨC GIANG")
    print("="*60)
    print(f"URL: http://localhost:{FLASK_CONFIG['port']}")
    print(f"API Chat: http://localhost:{FLASK_CONFIG['port']}/api/chat")
    print(f"API Stats: http://localhost:{FLASK_CONFIG['port']}/api/stats")
    print("="*60 + "\n")
    
    # Chạy Flask server
    app.run(
        host=FLASK_CONFIG['host'],
        port=FLASK_CONFIG['port'],
        debug=FLASK_CONFIG['debug'],
        threaded=FLASK_CONFIG['threaded']
    )
