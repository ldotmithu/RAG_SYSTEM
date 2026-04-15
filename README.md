# 🤖 AI - Production RAG Chatbot

A production-grade **Retrieval-Augmented Generation (RAG)** system built with modern AI technologies. Upload PDFs, ask questions, and get intelligent responses powered by Groq LLM and vector search.

<div align="center">

![React](https://img.shields.io/badge/React-18-61DAFB?style=flat&logo=react)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104-009688?style=flat&logo=fastapi)
![Groq](https://img.shields.io/badge/Groq-LLM-FF6B35?style=flat)
![Qdrant](https://img.shields.io/badge/Qdrant-Vector%20DB-FF1461?style=flat)

**[Features](#features) • [Quick Start](#quick-start) • [API Reference](#api-reference) • [Architecture](#architecture)**

</div>

---

## 📸  Demo:- 

![image](https://github.com/ldotmithu/Dataset/blob/main/rag_image.png)

---

## ✨ Features

- 📚 **Document Management** - Upload and index PDF documents with automatic chunking
- 🔍 **Semantic Search** - Find relevant content using 768-dimensional embeddings
- 💬 **Intelligent Chat** - Ask questions with Retrieval-Augmented Generation (RAG)
- ⚡ **Streaming Responses** - Real-time LLM responses for better UX
- 🎨 **Modern React UI** - Beautiful, responsive frontend
- 🏗️ **Production-Ready** - FastAPI backend with Groq API integration
- 🔒 **Privacy-Focused** - All processing local, API keys secure

---

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- Groq API Key (free: [console.groq.com](https://console.groq.com/keys))
- ~2GB disk space for models

### Installation

```bash
# Clone repository
git clone <your-repo-url>
cd beginner-local-rag-system

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set environment variable
export GROQ_API_KEY=gsk_your_key_here  # Windows: set GROQ_API_KEY=...
```

### Start the Application

```bash
# Terminal 1: Start backend
python run_backend.py
# → API running at http://localhost:8000
# → Docs at http://localhost:8000/docs

# Terminal 2: Start frontend
python run_frontend.py
# → UI running at http://localhost:3000
```

Open browser to **http://localhost:3000** and start chatting! 🎉

---

## 📋 API Reference

### Chat with Documents
```bash
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is the main topic?",
    "use_rag": true,
    "num_results": 5,
    "temperature": 0.7,
    "chat_history": []
  }'
```

### Upload Document
```bash
curl -X POST http://localhost:8000/api/documents/upload \
  -F "file=@document.pdf"
```

### List Documents
```bash
curl http://localhost:8000/api/documents
```

### Delete Document
```bash
curl -X DELETE http://localhost:8000/api/documents/document.pdf
```

### System Stats
```bash
curl http://localhost:8000/api/system/stats
```

Full API documentation available at: **http://localhost:8000/docs**

---

## 🏗️ Architecture

### Stack
| Component | Technology | Purpose |
|-----------|-----------|---------|
| **Frontend** | React 18 + Babel | Interactive UI |
| **Backend** | FastAPI + Uvicorn | High-performance async API |
| **RAG Pipeline** | Groq API | Direct LLM calls with vector search |
| **LLM** | Groq (latest) | Fast inference |
| **Embeddings** | sentence-transformers | 768-dimensional semantic vectors |
| **Vector DB** | Qdrant 1.17.1 | In-memory vector storage |

### Data Flow
```
📄 PDF Upload → Parse → Chunk (300 tokens) → Embed (768-dim) → Store (Qdrant)
              ↓
💬 User Query → Embed → Search Qdrant (Top-K) → Format Context → Groq LLM → Stream Response
```

---

## ⚙️ Configuration

### Environment Variables (`.env`)
```bash
GROQ_API_KEY=gsk_your_api_key_here
EMBEDDING_MODEL=sentence-transformers/all-mpnet-base-v2  # Optional
CHUNK_SIZE=300          # Optional
CHUNK_OVERLAP=100       # Optional
```

### Backend Settings (`src/constants.py`)
```python
GROQ_MODEL_NAME = "mixtral-8x7b-32768"  # Available: mixtral, llama2, gemma
EMBEDDING_DIMENSION = 768
```

### Frontend Customization (`frontend/static/styles.css`)
```css
--primary: #006d77;           /* Main color */
--secondary: #118ab2;         /* Accent */
--background: #f0f8ff;        /* Background */
```

---

## 📂 Project Structure

```
beginner-local-rag-system/
├── backend/
│   └── main.py                 # FastAPI app with all endpoints
├── frontend/
│   ├── index.html              # React entry point
│   └── static/
│       ├── app.jsx             # Main React component
│       └── styles.css          # Global styles
├── src/
│   ├── chat.py             # Chat interface with Groq API
│   ├── embeddings.py           # Model management
│   ├── ingestion.py            # Document processing
│   ├── qdrant_client.py        # Vector DB wrapper
│   ├── constants.py            # Configuration
│   └── utils.py                # Utilities
├── .env                        # Environment (git-ignored)
├── requirements.txt            # Dependencies
├── run_backend.py              # Backend startup
├── run_frontend.py             # Frontend startup
└── README.md                   # This file
```

---

## 🔧 Development

### Install New Packages
```bash
pip install <package-name>
pip freeze > requirements.txt
```

### Run Tests
```bash
pytest tests/
```

### Custom Chat Logic
Edit `src/chat.py`:
```python
def generate_response_streaming(query, use_hybrid_search=False, **kwargs):
    # Add custom Groq API logic here
    pass
```

### Add API Endpoints
Edit `backend/main.py`:
```python
@app.post("/api/custom")
async def custom_endpoint(data: Model):
    return {"result": "..."}
```

---

## 🐛 Troubleshooting

| Issue | Solution |
|-------|----------|
| "GROQ_API_KEY not set" | `export GROQ_API_KEY=gsk_...` |
| Slow model download | First run ~500MB, caches after (~/.cache/huggingface) |
| CORS errors | Ensure backend running on port 8000 |
| Low response quality | Increase `num_results` or lower `temperature` |
| "Module not found" | `pip install -r requirements.txt` |

---

## 📚 Learning Resources

- [Groq Documentation](https://console.groq.com/docs)
- [Groq API Docs](https://console.groq.com/docs)
- [Qdrant Vector DB](https://qdrant.tech/documentation)
- [FastAPI Tutorial](https://fastapi.tiangolo.com)
- [React Documentation](https://react.dev)

---

## 🤝 Contributing

1. Fork the repository
2. Create feature branch: `git checkout -b feature/awesome-feature`
3. Commit: `git commit -am 'Add awesome feature'`
4. Push: `git push origin feature/awesome-feature`
5. Create Pull Request

---

## 📄 License

MIT License - see [LICENSE](LICENSE) for details

---

## 🙏 Acknowledgments

Built with:
- ⚡ **Groq** - Ultra-fast LLM inference
- 🔗 **Groq** - Fast LLM inference API  
- 🎨 **React** - Modern UI framework
- 💾 **Qdrant** - Vector search engine
- 🚀 **FastAPI** - High-performance API framework

---

<div align="center">

  <h3>👨‍💻 Developed with by Mithurshan</h3>

  <p>
    🔗 <a href="https://www.linkedin.com/in/mithurshan6">LinkedIn</a> &nbsp;|&nbsp;
    💻 <a href="https://github.com/ldotmithu">GitHub</a>
  </p>

</div>

