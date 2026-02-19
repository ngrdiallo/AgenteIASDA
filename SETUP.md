# AgenteIA Clean Repository - Setup Instructions

## Location
📂 `C:\Users\A893apulia\Downloads\AgenteIASDA-Clean\`

## What's Included

### Core Files (9 Python modules)
- **app_gemini_server.py** - FastAPI server with WebSocket + REST + SSE log streaming
- **advanced_reasoning_llm.py** - Multi-model orchestration engine (85% complete)
- **config.py** - Configuration management
- **modalita_details.py** - Student learning modes
- **pdf_parser.py** - PDF extraction and processing
- **rag_engine.py** - Vector retrieval and embeddings
- **embedding_adapter.py** - Embedding model wrapper
- **vision_analysis_engine.py** - Image analysis capabilities
- **setup_and_run.py** - Automated setup script

### Frontend (Web UI)
- **templates/index.html** - Main HTML layout with log panel
- **static/app.js** - WebSocket client + SSE log streaming
- **static/style.css** - Gemini-style responsive design

### Configuration
- **.env.example** - Template for API keys (copy to .env)
- **requirements_PINNED.txt** - Exact dependency versions

### Documentation
- **README.md** - Complete project documentation
- **SETUP.md** - This file

### Directories
- **static/** - CSS and JS files
- **templates/** - HTML files
- **PDF/** - Sample documentation PDFs
- **logs/** - Server logs (created at runtime)
- **temp_uploads/** - Temporary file storage
- **storage/** - Data storage (created at runtime)

## Quick Start (3 steps)

### Step 1: Copy and Configure
```bash
cd AgenteIASDA-Clean
cp .env.example .env
# Edit .env and add your API keys:
# - GROQ_API_KEY (free, fast)
# - MISTRAL_API_KEY (free fallback)
# - HUGGINGFACE_API_KEY (optional)
```

### Step 2: Run Automated Setup
```bash
python setup_and_run.py
```
This will:
1. Create Python virtual environment
2. Install all dependencies
3. Validate .env file
4. Start server on http://localhost:8002

### Step 3: Open Browser
Navigate to `http://localhost:8002/`

## Manual Setup (Alternative)

```bash
# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements_PINNED.txt

# Run server
python app_gemini_server.py
```

## Features in This Clean Repo

✅ **4-Tier Orchestration** - Extractor→Analyzer→Enricher→Explainer
✅ **5-Tier Fallback** - Groq→Mistral→HF→Ollama Cloud→Ollama Local
✅ **Real-Time Log Streaming** - SSE-based browser log viewer
✅ **WebSocket Streaming** - Phase visibility during analysis
✅ **Quality Evaluation** - Confidence, depth, coverage metrics
✅ **Manual Escalation** - "Approfondisci" button
✅ **Multi-Format Support** - PDF, images, text
✅ **Knowledge Grounding** - RAG with embeddings

## What Was Excluded

❌ Test files (test_*.py) - 20 files
❌ Debug files (debug_*.py, run_debug_*.py) - 8 files
❌ Old versions (*_OLD.py) - 4 files
❌ Documentation files (except essential ones) - 15 .md files
❌ Log files (*.log, diagnostic outputs)
❌ Virtual environment (venv/)
❌ Cache files (__pycache__, *.pyc)
❌ Verification scripts (verify_*.py, health_check.py)

**Result**: Reduced from 150+ files to 36 essential files

## File Size

Total: ~250 KB (compressed ready for git)

## Next Steps

1. Initialize git repo
   ```bash
   git init
   git add .
   git commit -m "Initial commit: clean AgenteIA repository"
   ```

2. Push to remote (GitHub, GitLab, etc.)
   ```bash
   git remote add origin <your-repo-url>
   git push -u origin main
   ```

3. Deploy
   - Use `setup_and_run.py` on target machine
   - Or use Docker (create Dockerfile if needed)

## Support

See README.md for full documentation and API details.
