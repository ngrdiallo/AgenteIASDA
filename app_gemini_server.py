#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AgenteIA v5 - FastAPI Backend Server
Serves the Advanced Reasoning LLM with REST API + WebSocket for streaming
Paired with HTML/CSS/JS frontend for Gemini-style UI
"""

# LOAD ENVIRONMENT VARIABLES FIRST (before other imports)
from dotenv import load_dotenv
load_dotenv(verbose=True)

from fastapi import FastAPI, WebSocket, HTTPException, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel
from typing import Optional, List, Dict
import json
import asyncio
import logging
import os
import base64
import tempfile
import uuid
import glob
from pathlib import Path

# Import our LLM
from advanced_reasoning_llm import get_analysis_llm, ResponseQualityEvaluator, CognitiveStreamEmitter, escalate_response
from modalita_details import MODALITA_CONFIG

# Configure logging with streaming
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Log streaming system
from collections import deque
import threading

LOG_QUEUE = deque(maxlen=500)  # Keep last 500 logs
LOG_QUEUE_LOCK = threading.Lock()

class QueueHandler(logging.Handler):
    """Custom handler that sends logs to a thread-safe queue for streaming"""
    def emit(self, record):
        try:
            msg = self.format(record)
            with LOG_QUEUE_LOCK:
                LOG_QUEUE.append({
                    'timestamp': record.created,
                    'level': record.levelname,
                    'logger': record.name,
                    'message': msg,
                    'levelno': record.levelno
                })
        except Exception:
            self.handleError(record)

# Add queue handler to root logger
queue_handler = QueueHandler()
formatter = logging.Formatter('[%(asctime)s] %(levelname)s:%(name)s: %(message)s', datefmt='%H:%M:%S')
queue_handler.setFormatter(formatter)
logging.getLogger().addHandler(queue_handler)

# Initialize FastAPI
app = FastAPI(title="AgenteIA Enterprise", version="5.0")

# Increase WebSocket message size limit (default is 16MB, we allow up to 100MB)
from starlette.websockets import WebSocketState
import starlette.middleware
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.datastructures import Headers
import glob

# Service configuration
LLM_INSTANCE = None
# NOTE: No FILE_ID_MAP needed! File IDs are embedded in filenames: {uuid}_{originalname}
MODALITA_LIST = [
    ("generale", "Generale - Socrate.IA"),
    ("ragionamento", "Ragionamento - DeepSeek/Groq"),
    ("analisi", "Analisi - Gemini Vision"),
    ("fashion_design", "Fashion Design - Leonardo.IA"),
    ("esami", "Esami - Pascal.IA"),
    ("presentazioni", "Presentazioni - Spielberg.IA"),
    ("documenti", "Documenti - Tesla.IA")
]

# ============================================================
# PYDANTIC MODELS
# ============================================================
class ChatMessage(BaseModel):
    role: str  # "user" or "assistant"
    content: str
    timestamp: str
    backend: Optional[str] = None
    latency: Optional[str] = None

class QueryRequest(BaseModel):
    message: str
    modalita: str = "generale"
    forced_backend: Optional[str] = None
    file_base64: Optional[str] = None
    file_name: Optional[str] = None
    file_size: Optional[int] = None

class ModalitaInfo(BaseModel):
    key: str
    display_name: str
    icon: str
    nome_ia: str

# ============================================================
# WEBSOCKET CONTEXT FOR BACKEND ATTEMPTS
# ============================================================

class WebSocketContext:
    """Context to pass WebSocket reference to LLM callbacks"""
    def __init__(self):
        self.websocket = None
    
    async def send_attempt(self, attempt: int, total: int, backend_name: str):
        """Send backend attempt update to client"""
        if self.websocket:
            try:
                await self.websocket.send_json({
                    "type": "attempt",
                    "attempt": attempt,
                    "total": total,
                    "backend": backend_name,
                    "content": f"Tentativo {attempt}/{total}: {backend_name}..."
                })
            except Exception as e:
                logger.debug(f"Failed to send attempt: {e}")

# ============================================================
# FILE HANDLING
# ============================================================

def decode_and_save_file(file_base64: str, file_name: str) -> str:
    """Decode base64 file and save to temp directory. Return file path."""
    try:
        # Remove data URL prefix if present
        if ',' in file_base64:
            file_base64 = file_base64.split(',')[1]
        
        # Decode base64
        file_data = base64.b64decode(file_base64)
        
        # Create temp directory for files using ABSOLUTE path
        server_dir = os.path.dirname(os.path.abspath(__file__))
        temp_dir = os.path.join(server_dir, 'temp_uploads')
        os.makedirs(temp_dir, exist_ok=True)
        
        # Save file
        file_path = os.path.join(temp_dir, file_name)
        with open(file_path, 'wb') as f:
            f.write(file_data)
        
        logger.info(f'[FILE] Saved: {file_name} -> {file_path} (absolute)')
        return file_path
    except Exception as e:
        logger.error(f'[FILE] Error decoding file: {e}')
        return None


def load_file_context(file_path: str, max_chars: int = 8000) -> str:
    """Read textual content for orchestration context (best-effort)."""
    if not file_path or not os.path.exists(file_path):
        return ""

    ext = os.path.splitext(file_path)[1].lower()
    try:
        if ext in {'.txt', '.md', '.json', '.csv', '.log'}:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                return f.read()[:max_chars]
        if ext == '.pdf':
            from PyPDF2 import PdfReader
            reader = PdfReader(file_path)
            text = []
            for page in reader.pages:
                if len(''.join(text)) >= max_chars:
                    break
                page_text = page.extract_text() or ""
                text.append(page_text[: max_chars - len(''.join(text))])
            return "\n".join(text)
        # Unknown binary (image, pptx): return empty, orchestrator handles vision elsewhere
    except Exception as e:
        logger.debug(f"[FILE] Context read failed: {e}")
    return ""

# ============================================================
# STARTUP / SHUTDOWN
# ============================================================
@app.on_event("startup")
async def startup_event():
    global LLM_INSTANCE
    try:
        LLM_INSTANCE = get_analysis_llm()
        logger.info("[OK] LLM loaded successfully")
    except Exception as e:
        logger.error(f"[ERROR] Failed to load LLM: {e}")
        raise

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("[OK] AgenteIA server shutting down")

# ============================================================
# REST API ENDPOINTS
# ============================================================

@app.get("/api/health")
async def health_check():
    """Check if server and LLM are ready"""
    return {
        "status": "healthy",
        "llm_loaded": LLM_INSTANCE is not None,
        "version": "5.0"
    }
@app.get("/api/logs/stream")
async def stream_logs():
    """Stream server logs via Server-Sent Events (SSE)"""
    async def log_generator():
        # Send initial batch of existing logs
        with LOG_QUEUE_LOCK:
            for log_entry in list(LOG_QUEUE):
                yield f"data: {json.dumps(log_entry)}\n\n"
        
        # Stream new logs as they arrive
        last_index = len(LOG_QUEUE)
        while True:
            await asyncio.sleep(0.1)  # Poll every 100ms
            with LOG_QUEUE_LOCK:
                current_logs = list(LOG_QUEUE)
            
            # Send only new logs
            for log_entry in current_logs[last_index:]:
                yield f"data: {json.dumps(log_entry)}\n\n"
            
            last_index = len(current_logs)
    
    return StreamingResponse(log_generator(), media_type="text/event-stream")

@app.get("/api/logs/latest")
async def get_latest_logs(limit: int = 50):
    """Get latest N logs as JSON (non-streaming)"""
    with LOG_QUEUE_LOCK:
        logs = list(LOG_QUEUE)[-limit:]
    return {"logs": logs, "count": len(logs)}
@app.get("/api/modalita")
async def get_modalita():
    """Get list of available modalità with backend info"""
    result = {}
    for key in MODALITA_CONFIG:
        config = MODALITA_CONFIG[key]
        result[key] = {
            "nome_ia": config.get("nome_ia", "N/A"),
            "icona": config.get("icona", "🤖"),
            "titolo": config.get("titolo", "N/A"),
            "utilizzo": config.get("utilizzo", "N/A"),
            "backend_preferito": config.get("backend_preferito", "Auto"),
            "token_limit": config.get("token_limit", 2048),
            "specialita": config.get("specialita", [])
        }
    return result

@app.get("/api/backends")
async def get_backends():
    """Get list of available backends for forced selection"""
    if not LLM_INSTANCE:
        return {"error": "LLM not loaded"}
    
    backends = LLM_INSTANCE.get_available_backends()
    return {
        "backends": backends,
        "description": "Select a specific backend or leave empty for automatic fallback"
    }

@app.post("/api/upload-file")
async def upload_large_file(file: UploadFile = File(...)):
    """
    Upload large files (> 5MB) via multipart/form-data
    Returns a file_id to reference in WebSocket messages
    
    File ID is embedded in filename as: {uuid}_{originalname}
    This ensures file survives server restart - no in-memory map needed
    """
    try:
        # Generate unique file ID
        file_id = str(uuid.uuid4())
        
        # Create temp directory using ABSOLUTE path
        # This ensures files are accessible regardless of cwd
        server_dir = os.path.dirname(os.path.abspath(__file__))
        temp_dir = os.path.join(server_dir, 'temp_uploads')
        os.makedirs(temp_dir, exist_ok=True)
        
        # Save file with ID embedded in filename (filesystem as database)
        # Format: {uuid}_{originalname}
        file_path = os.path.join(temp_dir, f"{file_id}_{file.filename}")
        
        # Read and save file content
        contents = await file.read()
        with open(file_path, 'wb') as f:
            f.write(contents)
        
        file_size_mb = len(contents) / (1024 * 1024)
        logger.info(f'[REST-UPLOAD] File: {file.filename} ({file_size_mb:.2f} MB) → {file_id} @ {file_path}')
        
        return {
            "file_id": file_id,
            "file_name": file.filename,
            "file_size": len(contents),
            "file_size_mb": round(file_size_mb, 2),
            "message": f"File uploaded successfully. Use file_id '{file_id}' in chat"
        }
    except Exception as e:
        logger.error(f'[REST-UPLOAD] Error: {e}')
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/chat")
async def chat_completion(request: QueryRequest):
    """
    Send a message and get a synchronous response
    For streaming, use WebSocket endpoint instead
    """
    if not LLM_INSTANCE:
        raise HTTPException(status_code=503, detail="LLM not loaded")
    
    if not request.message.strip():
        raise HTTPException(status_code=400, detail="Empty message")
    
    try:
        # Call LLM with optional forced backend
        result = LLM_INSTANCE.complete(
            request.message.strip(),
            modalita=request.modalita,
            forced_backend=request.forced_backend
        )
        
        # Extract metadata
        latency = LLM_INSTANCE.last_response_metadata.get("latency", "?")
        backend = LLM_INSTANCE.active_backend or "unknown"
        
        return {
            "content": result.text,
            "backend": backend,
            "latency": latency,
            "modalita": request.modalita,
            "success": True
        }
    except Exception as e:
        logger.error(f"Chat error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================
# WEBSOCKET FOR STREAMING RESPONSES
# ============================================================
@app.websocket("/ws/chat")
async def websocket_chat(websocket: WebSocket):
    """
    WebSocket endpoint for streaming LLM responses with real-time backend attempt display
    Client sends: {"message": "...", "modalita": "...", "file_base64": "...", "file_name": "..."}
    Server streams: {"type": "attempt", ...}, {"type": "response", ...}, or {"type": "error", ...}
    """
    await websocket.accept()
    logger.info("[WS] Client connected")
    
    try:
        while True:
            # Receive message
            data = await websocket.receive_json()
            message = data.get("message", "").strip()
            modalita = data.get("modalita", "generale")
            forced_backend = data.get("forced_backend", None)
            
            # Handle both file upload strategies:
            file_base64 = data.get("file_base64")
            file_name = data.get("file_name")
            file_id = data.get("file_id")
            
            # DEBUG LOGGING
            logger.info(f"[WS-RECV] Message: {message[:30]}...")
            logger.info(f"[WS-RECV] file_base64: {'YES' if file_base64 else 'NO'}")
            logger.info(f"[WS-RECV] file_name: {file_name}")
            logger.info(f"[WS-RECV] file_id: {file_id}")
            
            if not message:
                await websocket.send_json({"type": "error", "content": "Empty message"})
                continue
            
            # Handle file from either source
            file_path = None
            
            # Priority 1: File ID (uploaded via REST)
            if file_id:
                logger.info(f"[WS-FILE] Attempting to find file with ID: {file_id}")
                # Search filesystem for file with this UUID
                server_dir = os.path.dirname(os.path.abspath(__file__))
                temp_dir = os.path.join(server_dir, 'temp_uploads')
                
                matches = glob.glob(os.path.join(temp_dir, f"{file_id}_*"))
                logger.info(f"[WS-FILE] Search in {temp_dir}: found {len(matches)} files")
                
                if matches:
                    file_path = matches[0]
                    file_name = os.path.basename(file_path).split('_', 1)[-1]
                    logger.info(f"[WS-FILE] SUCCESS: Found file -> {os.path.basename(file_path)}")
                else:
                    logger.warning(f"[WS-FILE] FAILED: No files found for {file_id}")
            
            # Priority 2: Inline base64 (small files)
            elif file_base64 and file_name:
                logger.info(f"[WS-FILE] Using inline base64 for file: {file_name}")
                file_path = decode_and_save_file(file_base64, file_name)
                if file_path:
                    logger.info(f"[WS-FILE] Decoded base64 -> {file_path}")
                else:
                    logger.warning(f"[WS-FILE] Failed to decode base64 for {file_name}")
            else:
                logger.info(f"[WS-FILE] No file detected")
            
            logger.info(f"[WS-FILE] Final file_path: {file_path if file_path else 'NONE'}")
            logger.info(f"[WS] Received: {message[:50]}... (modalita: {modalita})" + (f" [FILE: {file_name}]" if file_path else ""))
            
            # Create queue for real-time backend attempt notifications from executor thread
            attempt_queue: asyncio.Queue = asyncio.Queue()
            stream_queue: asyncio.Queue = asyncio.Queue()
            
            # Define callback that will be called from executor thread during LLM processing
            def on_attempt_callback(attempt_num: int, total: int, backend_name: str):
                """
                Called from executor thread when LLM tries each backend
                Uses put_nowait() to avoid blocking the executor thread
                """
                try:
                    attempt_queue.put_nowait({
                        "attempt": attempt_num,
                        "total": total,
                        "backend": backend_name,
                        "content": f"Tentativo {attempt_num}/{total}: {backend_name}..."
                    })
                    logger.info(f"[CALLBACK] Queued: Tentativo {attempt_num}/{total}: {backend_name}")
                except Exception as e:
                    logger.warning(f"[CALLBACK] Error queuing attempt: {e}")

            # Stream callback for orchestration phases
            def on_stream_callback(phase: str, content: str, model: str = None, data: Dict = None):
                try:
                    stream_queue.put_nowait({
                        "phase": phase,
                        "content": content,
                        "model": model,
                        "data": data or {}
                    })
                except Exception as e:
                    logger.debug(f"[STREAM] queue error: {e}")
            
            # Prepare LLM instance with callback
            LLM_INSTANCE.attempts_log = []
            LLM_INSTANCE.on_attempt = on_attempt_callback
            
            # Send thinking indicator
            await websocket.send_json({
                "type": "thinking",
                "content": "💭 Sto pensando..."
            })
            
            # Create task to process queue and send attempts in real-time
            queue_done = asyncio.Event()
            stream_done = asyncio.Event()
            
            async def send_queue_messages():
                """Process messages from attempt_queue and send to client"""
                logger.info("[QUEUE_TASK] Started processing queue")
                while not queue_done.is_set():
                    try:
                        # Wait for message with timeout
                        attempt = await asyncio.wait_for(attempt_queue.get(), timeout=0.2)
                        logger.info(f"[QUEUE_TASK] Got from queue: {attempt['content']}")
                        await websocket.send_json({
                            "type": "attempt",
                            "attempt": attempt["attempt"],
                            "total": attempt["total"],
                            "backend": attempt["backend"],
                            "content": attempt["content"]
                        })
                        logger.info(f"[QUEUE_TASK] Sent to client: {attempt['content']}")
                    except asyncio.TimeoutError:
                        # No message in queue, continue waiting
                        pass
                    except asyncio.CancelledError:
                        logger.info("[QUEUE_TASK] Cancelled")
                        break
                    except Exception as e:
                        logger.warning(f"[QUEUE_TASK] Error: {e}")
                        break
                logger.info("[QUEUE_TASK] Finished")

            async def send_stream_messages():
                """Process orchestration streaming events"""
                logger.info("[STREAM_TASK] Started")
                while not stream_done.is_set():
                    try:
                        evt = await asyncio.wait_for(stream_queue.get(), timeout=0.2)
                        payload = {
                            "type": evt.get("phase", "handoff"),
                            "content": evt.get("content", ""),
                            "model": evt.get("model"),
                            "data": evt.get("data", {})
                        }
                        await websocket.send_json(payload)
                    except asyncio.TimeoutError:
                        pass
                    except asyncio.CancelledError:
                        break
                    except Exception as e:
                        logger.debug(f"[STREAM_TASK] Error: {e}")
                        break
                logger.info("[STREAM_TASK] Finished")
            
            # Start queue processor task
            queue_task = asyncio.create_task(send_queue_messages())
            stream_task = asyncio.create_task(send_stream_messages())
            
            # Call LLM asynchronously (with or without file)
            try:
                loop = asyncio.get_event_loop()
                
                # Use orchestration pipeline by default; if forced_backend provided, use fallback chain
                file_context = load_file_context(file_path)

                if data.get("force_escalate"):
                    emitter = CognitiveStreamEmitter(websocket)
                    await emitter.emit("decision", "Escalation manuale richiesta dall'utente")
                    prev_resp = data.get("previous_response", "")
                    result_text = await escalate_response(
                        original_question=message,
                        previous_response=prev_resp,
                        previous_model="utente_precedente",
                        file_context=file_context,
                        query_function=LLM_INSTANCE._query_mistral,
                        emitter=emitter
                    )
                    result = type("obj", (), {"text": result_text})
                    was_manual_escalation = True
                    orchestrated_payload = {}
                elif forced_backend:
                    result = await loop.run_in_executor(
                        None,
                        lambda: LLM_INSTANCE.complete(message, modalita=modalita, forced_backend=forced_backend)
                    )
                    was_manual_escalation = False
                    orchestrated_payload = {}
                else:
                    def run_orch():
                        return LLM_INSTANCE.orchestrate_multimodel(
                            question=message,
                            file_context=file_context,
                            enable_enrichment=True,
                            emit_callback=on_stream_callback
                        )

                    orchestrated_payload = await loop.run_in_executor(None, run_orch)
                    class _Res:
                        def __init__(self, text):
                            self.text = text
                    result = _Res(orchestrated_payload.get("final_text", ""))
                    was_manual_escalation = False
                
                # Signal queue processor to stop
                queue_done.set()
                stream_done.set()
                await queue_task
                await stream_task
                
                # Get any remaining items in queue and send them
                while not attempt_queue.empty():
                    try:
                        attempt = attempt_queue.get_nowait()
                        await websocket.send_json({
                            "type": "attempt",
                            "attempt": attempt["attempt"],
                            "total": attempt["total"],
                            "backend": attempt["backend"],
                            "content": attempt["content"]
                        })
                        logger.info(f"[WS] Sent (final): {attempt['content']}")
                    except asyncio.QueueEmpty:
                        break
                
                # ===== NEW: QUALITY EVALUATION AND ESCALATION =====
                evaluator = ResponseQualityEvaluator()
                quality = evaluator.evaluate(result.text, message)

                first_model = LLM_INSTANCE.active_backend or "unknown"
                was_escalated = False
                can_escalate = not was_manual_escalation

                # Auto escalation only if not forced orchestration already done
                if quality.needs_escalation and not was_manual_escalation and not orchestrated_payload:
                    emitter = CognitiveStreamEmitter(websocket)
                    await emitter.emit("decision", "Qualita insufficiente. Escalation automatica al tier 2 (Mistral)...")
                    result.text = await escalate_response(
                        original_question=message,
                        previous_response=result.text,
                        previous_model=first_model,
                        file_context=file_context,
                        query_function=LLM_INSTANCE._query_mistral,
                        emitter=emitter
                    )
                    was_escalated = True
                    can_escalate = False

                latency = LLM_INSTANCE.last_response_metadata.get("latency", "?")
                backend = LLM_INSTANCE.active_backend or "unknown"

                await websocket.send_json({
                    "type": "response",
                    "content": result.text,
                    "backend": backend,
                    "latency": latency,
                    "success": True,
                    "was_escalated": was_escalated or was_manual_escalation,
                    "can_escalate": can_escalate,
                    "quality": quality.__dict__ if hasattr(quality, "__dict__") else quality,
                    "model_used": first_model,
                    "orchestrated": bool(orchestrated_payload)
                })

                logger.info(f"[WS] Response sent ({backend} in {latency})")
                logger.info(f"[WS] Escalation: {was_escalated} | Can escalate: {can_escalate}")
                
            except Exception as e:
                logger.error(f"[WS] Error: {e}")
                queue_done.set()
                stream_done.set()
                try:
                    await queue_task
                    await stream_task
                except:
                    pass
                await websocket.send_json({
                    "type": "error",
                    "content": f"Errore: {str(e)[:100]}"
                })
            finally:
                # Clean up file
                if file_path and os.path.exists(file_path):
                    try:
                        os.remove(file_path)
                        logger.info(f"[FILE] Cleaned up: {file_path}")
                    except Exception as e:
                        logger.warning(f"[FILE] Cleanup failed: {e}")
    
    except Exception as e:
        logger.error(f"[WS] Connection error: {e}")
    finally:
        logger.info("[WS] Client disconnected")

# ============================================================# MANUAL ESCALATION ENDPOINT
# ============================================================
@app.post("/api/deepen")
async def deepen_response(request: QueryRequest):
    """
    Manual escalation endpoint: User clicks "Approfondisci" button
    Takes previous response and escalates to Mistral/Gemini for deeper analysis
    
    Request:
    {
        "message": "original question",
        "previous_response": "shallow response text",
        "file_id": "xxx" or "file_base64": "...",
        "file_name": "document.pdf"
    }
    """
    if not LLM_INSTANCE:
        raise HTTPException(status_code=503, detail="LLM not loaded")
    
    if not request.message.strip():
        raise HTTPException(status_code=400, detail="Empty question")
    
    try:
        # Load file if provided
        file_path = None
        file_context = ""
        
        # Handle file like in WebSocket
        if request.file_id:
            server_dir = os.path.dirname(os.path.abspath(__file__))
            temp_dir = os.path.join(server_dir, 'temp_uploads')
            matches = glob.glob(os.path.join(temp_dir, f"{request.file_id}_*"))
            if matches:
                file_path = matches[0]
                file_context = load_file_context(file_path)
        elif request.file_base64 and request.file_name:
            file_path = decode_and_save_file(request.file_base64, request.file_name)
            file_context = load_file_context(file_path) if file_path else ""
        
        # Escalate using Mistral (escalate_response is async, must await)
        loop = asyncio.get_event_loop()
        escalated_text = await escalate_response(
            original_question=request.message,
            previous_response=request.message or "",
            previous_model="user_request",
            file_context=file_context,
            query_function=LLM_INSTANCE._query_mistral,
            emitter=None
        )
        
        latency = LLM_INSTANCE.last_response_metadata.get("latency", "?")
        backend = LLM_INSTANCE.active_backend or "Mistral"
        
        return {
            "content": escalated_text,
            "backend": backend,
            "latency": latency,
            "was_escalated": True,
            "model_used": "Mistral Escalation",
            "success": True
        }
        
    except Exception as e:
        logger.error(f"Deepen error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Cleanup file
        if file_path and os.path.exists(file_path):
            try:
                os.remove(file_path)
                logger.info(f"[DEEPEN] Cleaned up: {file_path}")
            except:
                pass

# ============================================================
# SERVE STATIC FILES AND HTML
# ============================================================
@app.get("/")
async def root():
    """Serve the main HTML page"""
    return FileResponse("templates/index.html", media_type="text/html")

# ============================================================
# VISION ANALYSIS & IMAGE GENERATION ENDPOINTS (NEW)
# ============================================================

@app.post("/api/image-analysis")
async def analyze_image(request: QueryRequest):
    """
    Analyze image artistically using Vision-Language models
    Supports: DeepSeek-VL2, Gemini Vision, HuggingFace Vision
    Best for: Artwork analysis, composition, style, techniques
    
    Request:
    {
        "message": "Analizza questo quadro dal punto di vista artistico",
        "file_base64": "<base64 encoded image>",
        "file_name": "artwork.jpg",
        "modalita": "analisi"  (optional, default "analisi")
    }
    """
    if not LLM_INSTANCE:
        raise HTTPException(status_code=503, detail="LLM not loaded")
    
    if not request.file_base64 or not request.file_name:
        raise HTTPException(status_code=400, detail="Missing image file")
    
    try:
        from vision_analysis_engine import get_vision_engine
        
        # Decode and save image
        file_path = decode_and_save_file(request.file_base64, request.file_name)
        if not file_path:
            raise HTTPException(status_code=400, detail="Failed to process image")
        
        # Analyze with Vision Engine
        vision_engine = get_vision_engine()
        
        analysis_depth = "standard"  # Could be "quick", "standard", "deep"
        if "profonda" in request.message.lower() or "estensiva" in request.message.lower():
            analysis_depth = "deep"
        elif "rapida" in request.message.lower():
            analysis_depth = "quick"
        
        result, metadata = vision_engine.analyze_artwork(
            file_path, 
            analysis_depth=analysis_depth
        )
        
        return {
            "content": result,
            "image_file": request.file_name,
            "analysis_depth": analysis_depth,
            "backend": metadata.get("backend", "unknown"),
            "latency": metadata.get("latency", "?"),
            "success": "ERROR" not in result
        }
        
    except Exception as e:
        logger.error(f"Image analysis error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/generate-art")
async def generate_artistic_image(request: QueryRequest):
    """
    Generate artistic images using Flux 2 Max (FREE)
    Completely free, no API key needed, direct image generation
    Best for: Creating artistic visualizations, design inspiration
    
    Request:
    {
        "message": "A serene Italian landscape with vineyards and Tuscan architecture, Renaissance style",
        "modalita": "creative" (optional)
    }
    """
    if not request.message.strip():
        raise HTTPException(status_code=400, detail="Empty prompt")
    
    try:
        from vision_analysis_engine import get_vision_engine
        
        # Extract style if mentioned in prompt
        style = "artistic"
        if "renaissance" in request.message.lower():
            style = "renaissance"
        elif "baroque" in request.message.lower():
            style = "baroque"
        elif "impressionist" in request.message.lower():
            style = "impressionist"
        elif "modern" in request.message.lower():
            style = "modern art"
        elif "digital" in request.message.lower():
            style = "digital art"
        
        # Generate image
        vision_engine = get_vision_engine()
        image_url, metadata = vision_engine.generate_artistic_image(
            request.message.strip(),
            style=style
        )
        
        if not image_url:
            raise HTTPException(status_code=500, detail="Image generation failed")
        
        return {
            "image_url": image_url,
            "prompt": request.message,
            "style": style,
            "generator": metadata.get("generator", "unknown"),
            "free_tier": metadata.get("free_tier", False),
            "latency": metadata.get("latency", "?"),
            "success": True
        }
        
    except Exception as e:
        logger.error(f"Image generation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Mount static files
try:
    app.mount("/static", StaticFiles(directory="static"), name="static")
except Exception as e:
    logger.warning(f"Could not mount static files: {e}")

# ============================================================
# RUN SERVER
# ============================================================
if __name__ == "__main__":
    import uvicorn
    
    # Run on localhost:8002
    print("\n[OK] AgenteIA v5 Enterprise - FastAPI Server")
    print("[OK] Server starting on http://localhost:8002")
    print("[OK] Open browser and navigate to location above\n")
    
    uvicorn.run(
        app,
        host="127.0.0.1",
        port=8002,
        log_level="info"
    )
