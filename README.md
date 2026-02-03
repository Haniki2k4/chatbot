# Chatbot Bệnh Viện Đức Giang

> Chatbot RAG (Retrieval-Augmented Generation) sử dụng BERT Embedding + LLM Local (Qwen2.5 / Llama)

## 📋 Mục lục

- [Tính năng](#-tính-năng)
- [Yêu cầu hệ thống](#-yêu-cầu-hệ-thống)
- [Cài đặt](#-cài-đặt)
- [Sử dụng](#-sử-dụng)
- [Cấu trúc dự án](#-cấu-trúc-dự-án)
- [API Documentation](#-api-documentation)
- [Cấu hình](#-cấu-hình-configsettingspy)
- [Troubleshooting](#️-troubleshooting)

## ✨ Tính năng

- ✅ Semantic search với BERT embeddings
- ✅ LLM local (Qwen2.5 1.5B hoặc Llama 3.2 1B)
- ✅ Ranking với xác suất (Softmax normalization)
- ✅ Web interface với Flask
- ✅ REST API cho chatbot
- ✅ Cache optimization (pickle)
- ✅ Response time tracking
- ✅ Vietnamese language support

## 🔧 Yêu cầu hệ thống

- **Python**: 3.8+
- **RAM**: >= 4GB
- **CPU**: 2+ cores (khuyến nghị)
- **Disk**: >= 2GB (cho models)

## 📦 Cài đặt

### 1. Clone / Setup repository
```bash
cd chatbot
```

### 2. Cài đặt dependencies
```bash
pip install -r requirements.txt
```

### 3. Chuẩn bị dữ liệu

**Tùy chọn A: Crawl dữ liệu tự động**
```bash
python scripts/download_model.py  # Download LLM models
```

**Tùy chọn B: Sử dụng dữ liệu có sẵn**
- Đặt các file `.txt` vào thư mục `data/raw/duc_giang_txt/`

### 4. Download LLM Models (tùy chọn)
```bash
python scripts/download_model.py
```

Hoặc download thủ công:
- [Qwen2.5 1.5B](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF)
- [Llama 3.2 1B](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct-GGUF)

Đặt vào: `data/models/`

## 🚀 Sử dụng

### Mode Web Interface (Recommended)
```bash
python web/app.py
```
Truy cập: `http://localhost:5000`

### Mode CLI
```bash
python scripts/run_cli.py
```

### Python API
```python
from src.chatbot_engine import DucGiangChatbot

# Khởi tạo chatbot
bot = DucGiangChatbot()

# Lấy câu trả lời
response = bot.get_response("Bệnh viện mở cửa lúc mấy giờ?")
print(response)

# Với chi tiết
response, scores, inference_time = bot.get_response(
    "Bệnh viện mở cửa lúc mấy giờ?",
    return_scores=True
)
print(f"Response: {response}")
print(f"Inference time: {inference_time:.2f}s")
```

## 📁 Cấu trúc dự án

```
chatbot/
├── README.md                      # Tài liệu này
├── requirements.txt               # Dependencies
├── .gitignore
│
├── config/                        # Cấu hình
│   ├── __init__.py
│   └── settings.py               # Cấu hình toàn cục
│
├── src/                           # Source code chính
│   ├── __init__.py
│   ├── chatbot_engine.py          # Engine chính
│   └── utils.py                   # Utility functions
│
├── data/                          # Dữ liệu
│   ├── raw/
│   │   └── duc_giang_txt/        # Text data
│   ├── cache/                     # Cache files
│   │   └── chatbot_cache.pkl
│   └── models/                    # LLM models
│       ├── qwen2.5-1.5b-instruct-q4_k_m.gguf
│       └── llama-3.2-1b-instruct-q4_k_m.gguf
│
├── web/                           # Web application
│   ├── app.py                     # Flask server
│   ├── templates/
│   │   └── index.html
│   └── static/
│       └── style.css
│
├── scripts/                       # Utility scripts
│   ├── run_cli.py                # CLI runner
│   ├── download_model.py         # Model downloader
│   └── setup.py
│
├── tests/                         # Unit tests (tương lai)
├── logs/                          # Log files
└── **/__pycache__/                # Python cache (auto-generated)
```

## 🔌 API Documentation

### POST /api/chat
**Gửi câu hỏi và nhận câu trả lời**

Request:
```json
{
  "message": "Bệnh viện có khoa nào?",
  "top_k": 5
}
```

Response:
```json
{
  "response": "Dựa trên thông tin...",
  "time": 2.34,
  "inference_time": 1.23,
  "scores": [
    {
      "rank": 1,
      "similarity": 0.95,
      "probability": 0.87,
      "text": "..."
    }
  ]
}
```

### GET /api/stats
**Lấy thống kê chatbot**

Response:
```json
{
  "total_chunks": 2500,
  "embedding_dim": 384,
  "model": "sentence-transformers/all-MiniLM-L6-v2",
  "llm_enabled": true,
  "llm_model": "data/models/qwen2.5-1.5b-instruct-q4_k_m.gguf"
}
```

### GET /api/health
**Health check**

Response:
```json
{
  "status": "ok",
  "message": "Chatbot is running"
}
```

## 🔧 Cấu hình (config/settings.py)

### CHATBOT_CONFIG
```python
{
    "data_folder": "data/raw/duc_giang_txt",
    "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
    "chunk_size": 3,
    "top_k": 5,
    "similarity_threshold": 0.3,
    "cache_file": "data/cache/chatbot_cache.pkl",
    "use_llm": True,
    "llm_model_path": "data/models/qwen2.5-1.5b-instruct-q4_k_m.gguf"
}
```

### LLM_CONFIG
```python
{
    "n_ctx": 512,           # Context window
    "n_threads": 4,         # Số threads
    "n_batch": 64,          # Batch size
    "temperature": 0.3,     # Độ sáng tạo
    "top_p": 0.9,          # Sampling
    "max_tokens": 200       # Token tối đa
}
```

## 🛠️ Troubleshooting

### 1. Lỗi: "llama-cpp-python chưa được cài đặt"
```bash
pip install llama-cpp-python
```

### 2. Lỗi: "Không tìm thấy model LLM"
- Download models từ HuggingFace
- Đặt vào `data/models/`
- Cập nhật đường dẫn trong `config/settings.py`

### 3. Lỗi: "Không có file txt"
- Đặt dữ liệu vào `data/raw/duc_giang_txt/`
- Hoặc chạy `python scripts/crawler.py`

### 4. Xóa cache cũ
```bash
rm data/cache/chatbot_cache.pkl
# Lần tiếp theo sẽ rebuild index tự động
```

---

**Last Updated**: February 2026
