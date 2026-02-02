# -*- coding: utf-8 -*-
"""
Flask Web Server cho Chatbot Bệnh viện Đức Giang
API endpoints cho frontend giao tiếp với chatbot
"""

from flask import Flask, render_template, request, jsonify
from chatbot_engine import DucGiangChatbot
import time
import os

app = Flask(__name__)

# Khởi tạo chatbot (chỉ khởi tạo 1 lần)
print("🚀 Đang khởi động server...")
chatbot = None

def init_chatbot():
    """Khởi tạo chatbot với hoặc không có LLM"""
    global chatbot
    if chatbot is None:
        # Kiểm tra xem có model LLM không
        llm_path = "models/llama-3.2-1b-instruct-q4_k_m.gguf"
        use_llm = os.path.exists(llm_path)
        
        if use_llm:
            print(f"🤖 Tìm thấy model LLM tại: {llm_path}")
            chatbot = DucGiangChatbot(
                data_folder="duc_giang_txt",
                llm_model_path=llm_path,
                use_llm=True
            )
        else:
            print("📝 Khởi tạo chatbot chỉ với BERT embeddings (không có LLM)")
            chatbot = DucGiangChatbot(data_folder="duc_giang_txt")
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
            "top_k": 3 (optional)
        }
    
    Response JSON:
        {
            "response": "câu trả lời",
            "time": thời gian xử lý (seconds),
            "scores": [thông tin xác suất]
        }
    """
    try:
        # Lấy dữ liệu từ request
        data = request.get_json()
        user_message = data.get('message', '').strip()
        top_k = data.get('top_k', 3)
        
        if not user_message:
            return jsonify({
                'error': 'Vui lòng nhập câu hỏi'
            }), 400
        
        # Đo thời gian xử lý
        start_time = time.time()
        
        # Lấy câu trả lời từ chatbot
        bot = init_chatbot()
        response, scores = bot.get_response(
            user_message,
            top_k=top_k,
            return_scores=True
        )
        
        end_time = time.time()
        processing_time = round(end_time - start_time, 2)
        
        # Trả về kết quả
        return jsonify({
            'response': response,
            'time': processing_time,
            'scores': scores[:2]  # Chỉ trả về top 2 scores
        })
    
    except Exception as e:
        print(f"❌ Lỗi: {str(e)}")
        return jsonify({
            'error': f'Đã xảy ra lỗi: {str(e)}'
        }), 500


@app.route('/api/stats', methods=['GET'])
def stats():
    """
    API endpoint lấy thống kê về chatbot
    
    Response JSON:
        {
            "total_chunks": số lượng chunks,
            "embedding_dim": chiều vector embedding,
            "model": tên model
        }
    """
    try:
        bot = init_chatbot()
        stats_data = bot.get_stats()
        return jsonify(stats_data)
    except Exception as e:
        return jsonify({
            'error': str(e)
        }), 500


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
    return jsonify({
        'error': 'Không tìm thấy endpoint'
    }), 404


@app.errorhandler(500)
def server_error(e):
    """Xử lý 500 error"""
    return jsonify({
        'error': 'Lỗi server'
    }), 500


if __name__ == '__main__':
    # Khởi tạo chatbot trước khi start server
    init_chatbot()
    
    print("\n" + "="*60)
    print("🌐 SERVER CHATBOT BỆNH VIỆN ĐỨC GIANG")
    print("="*60)
    print("📍 URL: http://localhost:5000")
    print("📍 API Chat: http://localhost:5000/api/chat")
    print("📍 API Stats: http://localhost:5000/api/stats")
    print("="*60 + "\n")
    
    # Chạy Flask server
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=False,  
        threaded=True
    )
