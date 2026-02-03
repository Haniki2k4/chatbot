# 🏥 Chatbot Bệnh Viện Đức Giang

Vietnamese hospital chatbot powered by **RAG (Retrieval-Augmented Generation)** with BERT embeddings and local LLM support.

## ✨ Features

- **Semantic Search**: BERT embeddings + cosine similarity
- **Hybrid Retrieval**: Semantic + keyword overlap scoring
- **Local LLM**: Qwen2.5 or Llama GGUF models (no API calls)
- **Fast**: Cached embeddings, optimized inference
- **Clean UI**: Modern Flask web interface
- **Production Ready**: Configurable, dockerizable, HF Spaces compatible

## 🎯 Tech Stack

| Component | Technology |
|-----------|------------|
| **Embedding** | Sentence-Transformers (all-MiniLM-L6-v2) |
| **Retrieval** | Cosine Similarity + Keyword Overlap |
| **LLM** | Llama-CPP (Qwen2.5 / Llama quantized) |
| **Backend** | Flask + Gunicorn |
| **Frontend** | HTML5 + Vanilla JS + CSS3 |

## 📋 Prerequisites

- Python 3.8+
- RAM: 4GB+ (8GB+ recommended for LLM)
- GPU: Optional (increases inference speed)

## 🚀 Quick Start

### 1. Clone & Setup

```bash
git clone https://github.com/yourusername/chatbot-duc-giang.git
cd chatbot-duc-giang

# Create virtual environment
python -m venv venv

# Activate
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Prepare Data

Place your `.txt` documents in `data/documents/`:

```bash
data/documents/
├── document1.txt
├── document2.txt
└── ...
```

### 3. Download LLM Model (Optional)

For LLM support, download a quantized model:

```bash
# Create models directory
mkdir -p models

# Download Qwen2.5 (recommended)
cd models
wget https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/qwen2.5-1.5b-instruct-q4_k_m.gguf
cd ..
```

Or disable LLM in `.env`:
```env
USE_LLM=False
```

### 4. Run Locally

```bash
# Development
python app.py

# Production
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

Access at: **http://localhost:5000**

## 📁 Project Structure

```
chatbot-duc-giang/
├── src/                          # Source code
│   ├── chatbot_engine.py         # Core engine
│   ├── embedding.py              # Embedding utilities
│   └── utils.py                  # Helpers
├── config/                       # Configuration
│   └── __init__.py
├── web/                          # Flask app
│   ├── app.py
│   ├── templates/
│   │   └── index.html
│   └── static/
│       ├── style.css
│       └── script.js
├── data/
│   ├── documents/                # Training data
│   └── cache/                    # Embeddings cache
├── models/                       # LLM models
├── tests/                        # Unit tests
├── requirements.txt
├── README.md
├── DEPLOY.md
├── .env.example
└── .gitignore
```

## 🎛️ Configuration

Copy `.env.example` to `.env`:

```bash
cp .env.example .env
```

Edit `.env`:

```env
# Flask
FLASK_ENV=production
FLASK_DEBUG=False
PORT=7860

# Embedding
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2

# LLM
USE_LLM=True
LLM_MODEL_PATH=./models/qwen2.5-1.5b-instruct-q4_k_m.gguf

# Retrieval
SIMILARITY_THRESHOLD=0.3
TOP_K=5
CHUNK_SIZE=3
```

## 🔄 How It Works

```
1. User Input
   ↓
2. Preprocessing (lowercase, remove special chars)
   ↓
3. Embedding (Sentence-Transformers)
   ↓
4. Semantic Search (Cosine Similarity)
   ↓
5. Keyword Matching (Optional boost)
   ↓
6. Hybrid Scoring (0.85 semantic + 0.15 keyword)
   ↓
7. Top-K Results Filtering
   ↓
8. LLM Response Generation (Optional)
   ↓
9. User Response
```

## 📊 API Endpoints

### POST `/api/chat`

```bash
curl -X POST http://localhost:5000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Bệnh viện có phòng khám ngoại tổng quát không?", "top_k": 5}'
```

**Response:**
```json
{
  "response": "Có, bệnh viện có phòng khám ngoại tổng quát...",
  "inference_time": 0.45,
  "scores": [
    {
      "rank": 1,
      "similarity": 0.82,
      "probability": 0.65,
      "text": "..."
    }
  ]
}
```

### GET `/api/stats`

```bash
curl http://localhost:5000/api/stats
```

**Response:**
```json
{
  "total_chunks": 1250,
  "embedding_dim": 384,
  "model": "sentence-transformers/all-MiniLM-L6-v2",
  "llm_enabled": true,
  "llm_model": "./models/qwen2.5-1.5b-instruct-q4_k_m.gguf"
}
```

### GET `/api/health`

Health check endpoint.

## 🐳 Docker Deployment

### Build Image

```bash
docker build -t chatbot-duc-giang .
```

### Run Container

```bash
docker run -p 5000:5000 \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/models:/app/models \
  chatbot-duc-giang
```

## 🤗 Hugging Face Spaces Deployment

See [DEPLOY.md](DEPLOY.md) for detailed instructions.

Quick start:

1. Fork repository to your GitHub
2. Go to [Hugging Face Spaces](https://huggingface.co/spaces)
3. Create new Space > Docker > Connect GitHub repo
4. Wait for deployment (~5-10 minutes)
5. Access at `https://huggingface.co/spaces/yourusername/chatbot-duc-giang`

## 📈 Performance

| Metric | Value |
|--------|-------|
| **Embedding Time** | ~50-100ms |
| **Search Time** | ~10-30ms |
| **LLM Inference** | ~200-500ms (depends on hardware) |
| **Total Latency** | ~300-700ms |
| **Throughput** | ~2-5 req/sec |

## 🛠️ Development

### Install Dev Dependencies

```bash
pip install -r requirements-dev.txt
```

### Run Tests

```bash
pytest tests/ -v
```

### Code Quality

```bash
# Format
black src/ tests/

# Lint
flake8 src/ tests/

# Sort imports
isort src/ tests/
```

## 🐛 Troubleshooting

### Issue: "Module not found"
**Solution:**
```bash
pip install -e .
```

### Issue: LLM fails to load
**Solution:**
- Check model path in `.env`
- Verify model file exists
- Ensure sufficient RAM (4GB+ recommended)

### Issue: Slow embeddings
**Solution:**
- Use GPU if available
- Reduce `CHUNK_SIZE`
- Enable cache: `chatbot_cache.pkl`

### Issue: Cache out of date
**Solution:**
```bash
rm data/cache/chatbot_cache.pkl
```

## 📝 License

MIT License - see LICENSE file

## 👥 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## 📧 Support

- **Issues**: GitHub Issues
- **Email**: your-email@example.com
- **Documentation**: [Wiki](https://github.com/yourusername/chatbot-duc-giang/wiki)

## 🙏 Acknowledgments

- [Sentence-Transformers](https://www.sbert.net/)
- [Llama-CPP](https://github.com/ggerganov/llama.cpp)
- [Flask](https://flask.palletsprojects.com/)
- [Qwen](https://qwenlm.github.io/)

---

**Made with ❤️ for Bệnh viện Đức Giang**
