# Chatbot Bệnh Viện Đức Giang - Hướng Dẫn Sử Dụng

## Tổng Quan

Chatbot RAG (Retrieval-Augmented Generation) sử dụng:
- **Embedding**: Sentence-Transformers (all-MiniLM-L6-v2)
- **Retrieval**: Cosine similarity + keyword overlap
- **LLM Local**: Qwen2.5 hoặc Llama (GGUF model)

## Yêu Cầu Hệ Thống

- Python 3.8+
- RAM: >= 4GB
- CPU: >= 2 cores (khuyến nghị)

## Cài Đặt

### 1. Cài đặt dependencies

```bash
pip install -r requirements.txt
```

File `requirements.txt`:
```
requests
beautifulsoup4
sentence-transformers
scikit-learn
numpy
llama-cpp-python
flask
```

### 2. Chuẩn bị dữ liệu

#### A: Crawl tự động

```bash
cd chatbot
python crawler.py
```

#### B: Sử dụng dữ liệu có sẵn

Đặt các file `.txt` vào thư mục `duc_giang_txt/`

### 3. Download LLM Model

```bash
python download_model.py
```

Chọn 1 trong 3 option:
1. Llama 3.2 1B (nhẹ, ~1.5GB)
2. Qwen 2.5 0.5B (siêu nhẹ, ~500MB)
3. Qwen 2.5 1.5B (tốt hơn, ~3GB)

**Output**: Model lưu vào `models/` (tự tạo)

## Chạy Chatbot

### Mode Terminal (tương tác)

```bash
python chatbot_engine.py
```

Nhập câu hỏi + Enter, gõ `exit` để thoát.


### Mode Web (Flask)

```bash
python app.py
```

Mở browser: `http://localhost:5000`

## 📁 Cấu Trúc Thư Mục

```
chatbot/
├── crawler.py              # Crawl dữ liệu từ website
├── chatbot_engine.py       # Engine chatbot 
├── app.py                  # Flask web server
├── download_model.py       # Download GGUF models
├── requirements.txt        # Dependencies
├── duc_giang_txt/          # Dữ liệu text 
├── models/                 # LLM models 
├── templates/
│   └── index.html          # Giao diện web
├── static/
│   └── style.css           # CSS styling
└── chatbot_cache.pkl       # Cache embeddings 
```

## Cấu Hình Tùy Chỉnh

### chatbot_engine.py

```python
bot = DucGiangChatbot(
    data_folder="duc_giang_txt", 
    model_name="sentence-transformers/all-MiniLM-L6-v2",  
    llm_model_path="models/qwen2.5-1.5b-instruct-q4_k_m.gguf", 
)
```
