# AgenteIA - Advanced Reasoning LLM for Art Students

Professional AI assistant for art students with multi-model orchestration, real-time streaming, and deep reasoning capabilities.

## Features

- **4-Tier Multi-Model Orchestration**: Extractor → Analyzer → Enricher → Explainer
- **5-Tier Fallback Chain**: Groq → Mistral → HuggingFace → Ollama Cloud → Ollama Local
- **Real-Time Log Streaming**: SSE-based log viewer with color-coded severity
- **WebSocket Streaming**: Real-time phase visibility during analysis
- **Quality Evaluation**: Confidence scoring, depth analysis, coverage metrics
- **Manual Escalation**: "Approfondisci" button for deeper analysis
- **Multi-Format Support**: PDF, images, text documents
- **Knowledge-Grounded**: RAG engine with vector embeddings

## Quick Start

### 1. Setup Environment

```bash
# Create virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements_PINNED.txt
```

### 2. Configure API Keys

Copy `.env.example` to `.env` and add your API keys:

```bash
cp .env.example .env
```

Required API keys:
- `GROQ_API_KEY` (fastest LLM, free tier available)
- `MISTRAL_API_KEY` (fallback analyzer, free tier available)
- `HUGGINGFACE_API_KEY` (second fallback)
- `GOOGLE_API_KEY` (optional, for Gemini enrichment)

### 3. Start Server

```bash
python app_gemini_server.py
```

Server starts on `http://localhost:8002`

### 4. Use in Browser

Navigate to `http://localhost:8002/` and:
1. Select modalità (mode) from sidebar
2. Upload PDF or paste text
3. Ask your question
4. Watch real-time phase streaming
5. Click "🔍 Log" button to see server logs in real-time

## Architecture

### Backend (Python/FastAPI)
- `app_gemini_server.py` - Main server with WebSocket + REST endpoints
- `advanced_reasoning_llm.py` - Multi-model orchestration engine
- `modalita_details.py` - Student learning modes configuration
- `pdf_parser.py` - PDF extraction and processing
- `rag_engine.py` - Vector embeddings and retrieval
- `embedding_adapter.py` - Embedding model integration

### Frontend (HTML/CSS/JavaScript)
- `templates/index.html` - Main UI layout
- `static/app.js` - WebSocket client and event handling
- `static/style.css` - Gemini-style responsive design

## API Endpoints

### WebSocket
- `GET /ws/chat` - Real-time streaming chat

### REST
- `GET /api/health` - Server health check
- `GET /api/modalita` - Available learning modes
- `GET /api/backends` - Available LLM backends
- `GET /api/logs/stream` - SSE log streaming
- `GET /api/logs/latest?limit=50` - Latest N logs as JSON
- `POST /api/chat` - Synchronous chat (non-streaming)
- `POST /api/deepen` - Manual escalation endpoint
- `POST /api/upload-file` - Large file upload (>5MB)
- `POST /api/image-analysis` - Vision model analysis

## Log Streaming

The system includes real-time log streaming:

1. Click "🔍 Log" button in header
2. Bottom panel opens with live server logs
3. Color-coded by severity: 🔵 INFO | 🟡 WARNING | 🔴 ERROR

Logs show:
- LLM API calls and response times
- File processing steps
- Quality evaluation results
- Escalation decisions

## Development

### Running Tests
```bash
# Integration test
python test_integration_rag.py

# Full system test
python test_production_dryrun.py
```

### Debugging
- All Python files have `logger.info()` calls that appear in log stream
- Browser console shows WebSocket messages
- Server logs in `logs/` directory

## Performance

- **Extraction**: 0.7-1.5s (Groq, ~1500 chars)
- **Analysis**: 1-3s (Mistral, deep reasoning)
- **Quality Eval**: <0.1s (local evaluation)
- **Total Response**: 1-4s per query

## Requirements

- Python 3.9+
- 2GB RAM minimum
- Internet connection (for cloud LLMs)
- Optional: CUDA for local LLM acceleration

## License

Private project - Closed source

## Support

Contact project maintainers for issues or feature requests.
