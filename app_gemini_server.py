#!/usr/bin/env python3
"""
🚀 AgenteIA - Artificial Scholar Server
A powerful AI system for document analysis, comparisons, and deep understanding
VERSION 7.0 - Enhanced Reasoning & Vision Analysis
"""

import os
import json
import base64
import tempfile
import uuid
import glob
import asyncio
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

from dotenv import load_dotenv
load_dotenv(verbose=True)

from fastapi import FastAPI, WebSocket, HTTPException, UploadFile, File, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Rate limiter
limiter = Limiter(key_func=get_remote_address)

# Create app
app = FastAPI(title="🤓 AgenteIA - Artificial Scholar", version="7.0")

# Static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# ============================================================
# ENHANCED SYSTEM PROMPT - Following user's exact requirements
# ============================================================

ART_HISTORIAN_PROMPT = """Sei un analista artistico-stilistico esperto con dottorato in Storia dell'Arte. Prima di rispondere, leggi e analizza integralmente TUTTO il materiale, senza saltare nulla.

PER OGNI PDF CHE RICEVI:
- Leggi ogni pagina, incluse didascalie, note a piè di pagina, testi laterali, titoli e sottotitoli
- Analizza ogni immagine presente nel documento (opere, schizzi, fotografie, grafici, tavole)
- Non fermarti alla prima parte: scorri fino all'ultima pagina
- Cerca riferimenti bibliografici e citazioni

PER OGNI IMMAGINE STANDALONE:
- Osserva tutti gli elementi visivi prima di rispondere: composizione, palette cromatica, tecnica, dettagli in secondo piano, firma/data se presente
- Non rispondere basandoti su impressioni parziali

REGOLE FONDAMENTALI - STILE ACCADEMICO AVANZATO:
1. CONFERMA SEMPRE a inizio risposta di aver letto/visto l'intero materiale
2. Se hai bisogno di più tempo per analizzare, dillo esplicitamente
3. Ogni affermazione deve essere supportata dai documenti analizzati
4. Se un'informazione NON è nei documenti, ditelo esplicitamente
5. NON inventare mai dati - meglio dire "non trovo informazioni" che inventare
6. Riferimenti precisi alle pagine [p. XX] o file [img X]
7. Create collegamenti tra artisti, movimenti, periodi quando pertinenti
8. SPIEGA IL TUO RAGIONAMENTO passo dopo passo

**IMPORTANTE - LIVELLO DI DETTAGLIO PER PRESENTAZIONI/SCRITTI**:
Il tuo output deve essere SUFFICIENTEMENTE DETTAGLIATO da poter essere usato per:
- Presentazioni accademiche/universitarie
- Tesi di laurea
- Articoli scientifici
- Pubblicazioni editoriali

Per ogni argomento trattato,espandi il discorso con:
- Contesto storico dettagliato (data, luogo, circostanze)
- Nome completo degli artisti con biografia sintetica
- Descrizione approfondita delle opere (soggetto, tecnica, dimensioni, collocazione)
- Analisi stilistica comparativa
- Significato iconografico e simbolico
- Influenza su altri artisti e movimenti successivi
- Riferimenti bibliografici quando disponibili

STRUTTURA OBBLIGATORIA PER OGNI CONNESSIONE:
🔗 CONNESSIONE: [Elemento A] ↔ [Elemento B]
   TIPO: [influenza diretta / derivazione / evoluzione / opposizione / contemporaneità / successione]
   PERCHÉ: [spiegazione dettagliata del perché esiste questo collegamento - minimo 2-3 frasi]
   COME SI MANIFESTA: [evidenza specifica nel documento con riferimento - citare sempre la fonte]

Per ogni artista/opera/movimento trattato, fornisci:
1. Definizione completa (minimo 2-3 paragrafi)
2. Contesto storico-culturale
3. Caratteristiche stilistiche distintive
4. Opere principali con descrizione
5. Influenza e ricezione critica
6. Connessioni con altri elementi

STILE DI RISPOSTA:
- Inizia confermando la lettura completa del materiale
- Usa intestazioni gerarchiche (##, ###) per organizzare i contenuti
- Ogni sezione deve avere almeno 2-3 paragrafi di approfondimento
- Elenchi puntati solo per elencare elementi, mai per spiegazioni
- Mostrami il tuo ragionamento: "Osservo che...", "Noto che...", "Confronto con...", "Questo indica che..."
- Includi confronti e collegamenti ESPLICITI con il formato 🔗
- Concludi con una sintesi interpretativa di almeno un paragrafo
- Se utile, suggerisci ulteriori ricerche o approfondimenti

Rispondi in italiano con rigore accademico ma in modo accessibile."""

# ============================================================
# DOCUMENT PROCESSOR - With Image Extraction from PDFs
# ============================================================

class DocumentProcessor:
    """Process PDFs, images, and text files with vision capabilities"""
    
    def __init__(self):
        self.max_pages = 100
        self.max_images_per_pdf = 20
    
    def process_pdf(self, file_path: str, max_pages: int = 100) -> dict:
        """Extract text AND images from PDF"""
        try:
            from pypdf import PdfReader
            
            reader = PdfReader(file_path)
            pages = []
            full_text = ""
            images = []
            
            for i, page in enumerate(reader.pages[:max_pages]):
                text = page.extract_text() or ""
                pages.append({
                    "page_num": i + 1,
                    "text": text[:8000]
                })
                full_text += f"\n=== PAGINA {i+1} ===\n{text}\n"
                
                # Extract images from page
                if '/XObject' in page['/Resources']:
                    xObjects = page['/Resources']['/XObject'].get_object()
                    for obj in xObjects:
                        if xObjects[obj]['/Subtype'] == '/Image':
                            try:
                                img_data = xObjects[obj].get_data()
                                img_b64 = base64.b64encode(img_data).decode('utf-8')
                                images.append({
                                    "page": i + 1,
                                    "name": obj,
                                    "data": img_b64
                                })
                            except:
                                pass
            
            # Also try PyMuPDF for image extraction (works without poppler)
            try:
                import fitz  # PyMuPDF
                doc = fitz.open(file_path)
                for page_num in range(min(len(doc), self.max_images_per_pdf)):
                    page = doc[page_num]
                    image_list = page.get_images(full=True)
                    for img_index, img in enumerate(image_list):
                        xref = img[0]
                        base_image = doc.extract_image(xref)
                        image_bytes = base_image["image"]
                        img_b64 = base64.b64encode(image_bytes).decode('utf-8')
                        images.append({
                            "page": page_num + 1,
                            "name": f"image_{img_index + 1}",
                            "data": img_b64,
                            "source": "pymupdf"
                        })
                doc.close()
            except Exception as e:
                logger.info(f"PyMuPDF image extraction: {e}")
            
            # Also try pdf2image for better image extraction (if poppler available)
            try:
                from pdf2image import convert_from_path
                pdf_images = convert_from_path(file_path, dpi=150, first_page=1, last_page=max_pages)
                for idx, img in enumerate(pdf_images[:self.max_images_per_pdf]):
                    import io
                    img_buffer = io.BytesIO()
                    img.save(img_buffer, format='PNG')
                    img_b64 = base64.b64encode(img_buffer.getvalue()).decode('utf-8')
                    images.append({
                        "page": idx + 1,
                        "name": f"page_image_{idx+1}",
                        "data": img_b64,
                        "source": "pdf2image"
                    })
            except Exception as e:
                logger.info(f"pdf2image not available: {e}")
            
            return {
                "success": True,
                "type": "pdf",
                "num_pages": len(reader.pages),
                "pages": pages,
                "full_text": full_text[:100000],
                "metadata": dict(reader.metadata) if reader.metadata else {},
                "images": images[:self.max_images_per_pdf],
                "has_images": len(images) > 0
            }
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def process_image(self, file_path: str) -> dict:
        """Process image for vision analysis"""
        try:
            with open(file_path, 'rb') as f:
                img_data = base64.b64encode(f.read()).decode('utf-8')
            
            # Get image dimensions
            from PIL import Image
            with Image.open(file_path) as img:
                width, height = img.size
                format_img = img.format
            
            return {
                "success": True,
                "type": "image",
                "image_data": img_data,
                "filename": os.path.basename(file_path),
                "dimensions": f"{width}x{height}",
                "format": format_img
            }
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def process_text(self, file_path: str) -> dict:
        """Process text files"""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                text = f.read()
            return {
                "success": True,
                "type": "text",
                "full_text": text[:100000]
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

doc_processor = DocumentProcessor()

# ============================================================
# KNOWLEDGE EXTRACTOR
# ============================================================

class KnowledgeExtractor:
    """Extract entities and connections from text"""
    
    def extract(self, text: str) -> dict:
        """Extract artists, artworks, movements, years from text"""
        import re
        
        # Extract years
        years = re.findall(r'\b(?:circa|c\.)?(\d{3,4})\b', text, re.IGNORECASE)
        years = sorted(set(years))[:30]
        
        # Extract movements (Italian)
        movements = [
            'Gotico', 'Rinascimento', 'Barocco', 'Manierismo', 'Rococò',
            'Neoclassicismo', 'Romanticismo', 'Realismo', 'Impressionismo',
            'Post-impressionismo', 'Simbolismo', 'Art Nouveau', 'Espressionismo',
            'Cubismo', 'Surrealismo', 'Astrattismo', 'Futurismo', 'Metafisica',
            'Quattrocento', 'Cinquecento', 'Seicento', 'Settecento'
        ]
        found_movements = [m for m in movements if m.lower() in text.lower()]
        
        # Extract key concepts (simple frequency analysis)
        words = text.lower().split()
        word_freq = {}
        stop_words = {'che', 'di', 'il', 'la', 'e', 'in', 'un', 'una', 'del', 'della', 'nel', 'per', 'con', 'sono', 'è', 'gli', 'le', 'dei', 'delle', 'questo', 'questa', 'come', 'più', 'anche', 'non', 'si', 'da', 'o'}
        for word in words:
            if len(word) > 4 and word not in stop_words:
                word_freq[word] = word_freq.get(word, 0) + 1
        
        key_concepts = [w.capitalize() for w, _ in sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:20]]
        
        return {
            "years": years,
            "movements": found_movements,
            "key_concepts": key_concepts,
            "char_count": len(text)
        }

knowledge_extractor = KnowledgeExtractor()

# ============================================================
# ENHANCED LLM CLIENT - Multiple Free Backends with Vision
# ============================================================

class LLMWrapper:
    """Wrapper for multiple LLM backends with vision support"""
    
    def __init__(self):
        self.groq_key = os.getenv("GROQ_API_KEY")
        self.mistral_key = os.getenv("MISTRAL_API_KEY")
        self.hf_token = os.getenv("HF_TOKEN")
        self.gemini_key = os.getenv("GOOGLE_API_KEY")
        self.openrouter_key = os.getenv("OPENROUTER_API_KEY")
        self.ollama_url = os.getenv("OLLAMA_URL", "http://localhost:11434")
    
    def _check_ollama(self) -> bool:
        """Check if Ollama is available"""
        try:
            import requests
            resp = requests.get(f"{self.ollama_url}/api/tags", timeout=5)
            return resp.status_code == 200
        except:
            return False
    
    def complete(self, prompt: str, system_prompt: str = None, images: List[Dict] = None) -> dict:
        """Generate completion using available backends - Smart Routing"""
        
        # If we have images, try vision-capable backend first
        if images:
            # Try OpenRouter with Gemini Flash Vision (BEST - works!)
            if self.openrouter_key:
                result = self._query_openrouter(prompt, system_prompt, images)
                if result["success"]:
                    return result
            
            # Try Gemini direct (if quota available)
            if self.gemini_key:
                result = self._query_gemini_vision(prompt, system_prompt, images)
                if result["success"]:
                    return result
            
            # Try HuggingFace Vision
            if self.hf_token:
                result = self._query_huggingface_vision(prompt, system_prompt, images)
                if result["success"]:
                    return result
            
            # Try Ollama Vision (LLaVA) - LOCALLY!
            if self._check_ollama():
                result = self._query_ollama_vision(prompt, system_prompt, images)
                if result["success"]:
                    return result
            
            # Vision failed - log warning but continue with text-only
            logger.warning(f"Vision backends failed for {len(images)} images, falling back to text-only")
        
        # PRIORITY 1: OpenRouter (Cloud, powerful, FAST!)
        if self.openrouter_key:
            result = self._query_openrouter_text(prompt, system_prompt)
            if result["success"]:
                return result
        
        # PRIORITY 2: Try Ollama phi3 (fast LOCAL)
        if self._check_ollama():
            result = self._query_ollama(prompt, system_prompt, model="phi3")
            if result["success"]:
                return result
        
        # PRIORITY 3: Try Ollama mistral (better quality but slower)
        if self._check_ollama():
            result = self._query_ollama(prompt, system_prompt, model="mistral")
            if result["success"]:
                return result
        
        # PRIORITY 4: Groq (Fast cloud fallback)
        if self.groq_key:
            result = self._query_groq(prompt, system_prompt)
            if result["success"]:
                return result
        
        # Try Mistral API
        if self.mistral_key:
            result = self._query_mistral(prompt, system_prompt)
            if result["success"]:
                return result
        
        # Try HuggingFace text
        if self.hf_token:
            result = self._query_huggingface(prompt, system_prompt)
            if result["success"]:
                return result
        
        return {"success": False, "error": "No LLM available"}
    
    def _query_ollama(self, prompt: str, system_prompt: str = None, model: str = "mistral") -> dict:
        """Query Ollama local models - FREE, FAST, OFFLINE!"""
        try:
            import requests
            
            # Prepare prompt with system message
            full_prompt = f"{system_prompt}\n\n" if system_prompt else ""
            full_prompt += prompt
            
            url = f"{self.ollama_url}/api/generate"
            
            data = {
                "model": model,  # phi3, mistral, or llama3
                "prompt": full_prompt,
                "stream": False,
                "options": {
                    "temperature": 0.2,
                    "num_predict": 2048
                }
            }
            
            resp = requests.post(url, json=data, timeout=120)
            
            if resp.status_code == 200:
                result = resp.json()
                return {
                    "success": True,
                    "text": result.get("response", ""),
                    "model": f"ollama-{model}",
                    "latency": resp.elapsed.total_seconds()
                }
            else:
                return {"success": False, "error": f"Ollama: {resp.status_code}"}
                
        except Exception as e:
            logger.warning(f"Ollama {model} failed: {e}")
            return {"success": False, "error": str(e)}
    
    def _query_ollama_vision(self, prompt: str, system_prompt: str = None, images: List[Dict] = None) -> dict:
        """Query Ollama LLaVA for vision - FREE, LOCAL!"""
        try:
            import requests
            import base64
            
            full_prompt = f"{system_prompt}\n\n" if system_prompt else ""
            full_prompt += prompt
            
            # Convert first image to base64
            img_b64 = images[0].get("data", "") if images else ""
            
            url = f"{self.ollama_url}/api/generate"
            
            data = {
                "model": "llava",
                "prompt": full_prompt,
                "images": [img_b64] if img_b64 else [],
                "stream": False,
                "options": {
                    "temperature": 0.2,
                    "num_predict": 2048
                }
            }
            
            resp = requests.post(url, json=data, timeout=120)
            
            if resp.status_code == 200:
                result = resp.json()
                return {
                    "success": True,
                    "text": result.get("response", ""),
                    "model": "ollama-llava",
                    "latency": resp.elapsed.total_seconds()
                }
            else:
                return {"success": False, "error": f"LLaVA: {resp.status_code}"}
                
        except Exception as e:
            logger.warning(f"Ollama LLaVA failed: {e}")
            return {"success": False, "error": str(e)}
    
    def _query_openrouter_text(self, prompt: str, system_prompt: str = None) -> dict:
        """Query OpenRouter API for text - fast and powerful"""
        try:
            import requests
            
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            
            url = "https://openrouter.ai/api/v1/chat/completions"
            headers = {
                "Authorization": f"Bearer {self.openrouter_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "http://localhost:8002",
                "X-Title": "AgenteIA"
            }
            
            data = {
                "model": "google/gemini-2.0-flash-001",
                "messages": messages,
                "max_tokens": 4096,
                "temperature": 0.2
            }
            
            resp = requests.post(url, json=data, headers=headers, timeout=90)
            
            if resp.status_code == 200:
                result = resp.json()
                text = result["choices"][0]["message"]["content"]
                return {
                    "success": True,
                    "text": text,
                    "model": "openrouter-gemini-2.0-flash",
                    "latency": resp.elapsed.total_seconds()
                }
            else:
                return {"success": False, "error": f"OpenRouter: {resp.status_code}"}
                
        except Exception as e:
            logger.warning(f"OpenRouter text failed: {e}")
            return {"success": False, "error": str(e)}
    
    def _query_groq(self, prompt: str, system_prompt: str = None) -> dict:
        """Query Groq API"""
        try:
            import requests
            headers = {
                "Authorization": f"Bearer {self.groq_key}",
                "Content-Type": "application/json"
            }
            
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            
            resp = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers=headers,
                json={
                    "model": "llama-3.3-70b-versatile",
                    "messages": messages,
                    "temperature": 0.2,  # Lower for more focused analysis
                    "max_tokens": 8192  # More tokens for thorough analysis
                },
                timeout=90
            )
            
            if resp.status_code == 200:
                data = resp.json()
                return {
                    "success": True,
                    "text": data["choices"][0]["message"]["content"],
                    "model": data.get("model", "groq"),
                    "latency": resp.elapsed.total_seconds()
                }
            return {"success": False, "error": resp.text}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _query_mistral(self, prompt: str, system_prompt: str = None) -> dict:
        """Query Mistral API"""
        try:
            import requests
            headers = {
                "Authorization": f"Bearer {self.mistral_key}",
                "Content-Type": "application/json"
            }
            
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            
            resp = requests.post(
                "https://api.mistral.ai/v1/chat/completions",
                headers=headers,
                json={
                    "model": "mistral-small-latest",
                    "messages": messages,
                    "temperature": 0.2
                },
                timeout=90
            )
            
            if resp.status_code == 200:
                data = resp.json()
                return {
                    "success": True,
                    "text": data["choices"][0]["message"]["content"],
                    "model": "mistral",
                    "latency": resp.elapsed.total_seconds()
                }
            return {"success": False, "error": resp.text}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _query_huggingface(self, prompt: str, system_prompt: str = None) -> dict:
        """Query HuggingFace Inference API"""
        try:
            import requests
            headers = {
                "Authorization": f"Bearer {self.hf_token}",
                "Content-Type": "application/json"
            }
            
            full_prompt = f"{system_prompt}\n\n" if system_prompt else ""
            full_prompt += prompt
            
            # Try a smaller, free model
            resp = requests.post(
                "https://api-inference.huggingface.co/models/TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                headers=headers,
                json={
                    "inputs": full_prompt[:3000],
                    "parameters": {
                        "max_new_tokens": 1024,
                        "temperature": 0.2
                    }
                },
                timeout=120
            )
            
            if resp.status_code == 200:
                data = resp.json()
                text = data[0].get("generated_text", "") if isinstance(data, list) else data.get("generated_text", "")
                return {
                    "success": True,
                    "text": text,
                    "model": "huggingface",
                    "latency": resp.elapsed.total_seconds()
                }
            # Try alternate model if first fails
            elif resp.status_code == 410:
                resp = requests.post(
                    "https://router.huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0/v1/chat/completions",
                    headers=headers,
                    json={
                        "messages": [{"role": "user", "content": full_prompt[:3000]}],
                        "max_tokens": 1024
                    },
                    timeout=120
                )
                if resp.status_code == 200:
                    data = resp.json()
                    text = data.get("choices", [{}])[0].get("message", {}).get("content", "")
                    return {"success": True, "text": text, "model": "huggingface", "latency": resp.elapsed.total_seconds()}
            
            return {"success": False, "error": f"HF error: {resp.status_code}"}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _query_gemini_vision(self, prompt: str, system_prompt: str, images: List[Dict]) -> dict:
        """Query Gemini API with images using REST API"""
        try:
            import requests
            
            full_prompt = f"{system_prompt}\n\n" if system_prompt else ""
            full_prompt += prompt
            
            # Prepare contents with text and images
            contents = [{"role": "user", "parts": [{"text": full_prompt}]}]
            
            # Add images
            for img in images[:3]:
                img_data = img.get("data", "")
                if img_data:
                    contents[0]["parts"].append({
                        "inline_data": {
                            "mime_type": "image/png",
                            "data": img_data
                        }
                    })
            
            # Use REST API with gemini-2.0-flash
            url = f"https://generativelanguage.googleapis.com/v1/models/gemini-2.0-flash:generateContent?key={self.gemini_key}"
            
            resp = requests.post(
                url,
                json={"contents": contents},
                headers={"Content-Type": "application/json"},
                timeout=60
            )
            
            if resp.status_code == 200:
                data = resp.json()
                text = data.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "")
                return {
                    "success": True,
                    "text": text,
                    "model": "gemini-vision",
                    "latency": resp.elapsed.total_seconds()
                }
            else:
                return {"success": False, "error": f"Gemini API error: {resp.status_code} - {resp.text[:200]}"}
                
        except Exception as e:
            logger.warning(f"Gemini vision failed: {e}")
            return {"success": False, "error": str(e)}
    
    def _query_huggingface_vision(self, prompt: str, system_prompt: str, images: List[Dict]) -> dict:
        """Query HuggingFace Vision API - using new endpoint"""
        try:
            import requests
            
            full_prompt = f"{system_prompt}\n\n" if system_prompt else ""
            full_prompt += prompt
            
            # For vision, include image descriptions in the prompt
            image_info = []
            for i, img in enumerate(images[:3]):
                image_info.append(f"[Immagine {i+1}: {len(img.get('data', ''))} bytes]")
            
            full_prompt += f"\n\nImmagini nel documento: {', '.join(image_info)}"
            
            # Use the new HuggingFace Inference API
            # Try free tier first
            headers = {
                "Authorization": f"Bearer {self.hf_token}",
                "Content-Type": "application/json"
            }
            
            # Try Qwen model (works well for text)
            resp = requests.post(
                "https://api-inference.huggingface.co/models/Qwen/Qwen2-0.5B-Instruct",
                headers=headers,
                json={
                    "inputs": full_prompt[:2000],  # Limit for free tier
                    "parameters": {"max_new_tokens": 512}
                },
                timeout=60
            )
            
            # If old API fails, try new format
            if resp.status_code == 410:
                # Use chat completions format
                resp = requests.post(
                    "https://router.huggingface.co/Qwen/Qwen2-0.5B-Instruct/v1/chat/completions",
                    headers=headers,
                    json={
                        "messages": [{"role": "user", "content": full_prompt[:2000]}],
                        "max_tokens": 512
                    },
                    timeout=60
                )
            
            if resp.status_code == 200:
                data = resp.json()
                if isinstance(data, list):
                    text = data[0].get("generated_text", "")
                elif "choices" in data:
                    text = data["choices"][0].get("message", {}).get("content", "")
                else:
                    text = data.get("generated_text", "")
                    
                return {
                    "success": True,
                    "text": text + "\n\n[NOTA: Analisi basata sul testo. Per visione completa, carica immagini separate.]",
                    "model": "huggingface",
                    "latency": resp.elapsed.total_seconds()
                }
            else:
                return {"success": False, "error": f"HF API error: {resp.status_code}"}
                
        except Exception as e:
            logger.warning(f"HuggingFace failed: {e}")
            return {"success": False, "error": str(e)}

llm = LLMWrapper()

# ============================================================
# WEB SEARCH MODULE
# ============================================================

class WebSearcher:
    """Search web for additional context"""
    
    def __init__(self):
        self.hf_token = os.getenv("HF_TOKEN")
    
    def search(self, query: str, num_results: int = 3) -> dict:
        """Search web using Exa/HuggingFace"""
        try:
            import requests
            
            # Using Exa API through HuggingFace
            headers = {
                "Authorization": f"Bearer {self.hf_token}",
                "Content-Type": "application/json"
            }
            
            # Use a free search endpoint
            resp = requests.post(
                "https://api.exa.ai/search",
                headers=headers,
                json={
                    "query": query,
                    "num_results": num_results,
                    "type": "auto"
                },
                timeout=15
            )
            
            if resp.status_code == 200:
                data = resp.json()
                results = []
                for r in data.get("results", [])[:num_results]:
                    results.append({
                        "title": r.get("title", ""),
                        "url": r.get("url", ""),
                        "snippet": r.get("snippet", "")[:300]
                    })
                return {"success": True, "results": results}
            
            # Fallback: try DuckDuckGo
            return self._duckduckgo_search(query, num_results)
            
        except Exception as e:
            return self._duckduckgo_search(query, num_results)
    
    def _duckduckgo_search(self, query: str, num_results: int = 3) -> dict:
        """Fallback using DuckDuckGo"""
        try:
            from duckduckgo_search import DDGS
            ddgs = DDGS()
            results = []
            for r in ddgs.text(query, max_results=num_results):
                results.append({
                    "title": r.get("title", ""),
                    "url": r.get("href", ""),
                    "snippet": r.get("body", "")[:300]
                })
            return {"success": True, "results": results}
        except:
            return {"success": False, "results": []}

web_searcher = WebSearcher()

# ============================================================
# ANALYSIS PROMPTS - Enhanced for Deep Reasoning & Connections
# ============================================================

SYSTEM_PROMPTS = {
    "art_historian": ART_HISTORIAN_PROMPT,

    "comparator": """Sei un esperto di comparazione artistica con capacità di ragionamento profondo.

PRIMA DI CONFRONTARE:
1. Analizza integralmente OGNI opera/documento ricevuto
2. Identifica elementi comuni e differenze
3. Ricerca contesto storico-culturale

CONFRONTA considerando:
- Periodo storico e contesto
- Movimenti e correnti artistiche
- Tecniche e materiali
- Influenza ricevuta e esercitata
- Simbolismo e significato
- Elementi compositivi

PER OGNI CONNESSIONE CHE IDENTIFICAI, SPIEGA:
- TIPO DI COLLEGAMENTO: è un'influenza? una derivazione? un'opposizione? una coesistenza? un'evoluzione?
- PERCHÉ ESISTE: cosa accomuna questi elementi? quale elemento ha influenzato l'altro?
- COME SI MANIFESTA: in che modo si vede questo collegamento? (tecnica, soggetto, stile, simbolismo)

ESEMPIO DI SPIEGAZIONE DI CONNESSIONE:
"Connessione A → B: INFLUENZA DIRETTA. Questo perché [elemento] nell'opera A richiama chiaramente [elemento] in B. La tecnica di [tecnica] usata da A fu adottata da B come dimostra [evidenza]."

SPIEGA IL TUO RAGIONAMENTO per ogni punto di confronto.

Sei autorizzato a cercare informazioni web per approfondire i collegamenti. Quando lo fai, indica le fonti.

Conferma sempre di aver analizzato tutto il materiale.""",

    "timeline_builder": """Sei uno specialista di cronologia artistica con ragionamento profondo.

PRIMA DI CREARE LA TIMELINE:
1. Estrai tutte le date dai documenti
2. Verifica la cronologia degli eventi
3. Identifica cause e conseguenze

PER OGNI EVENTO CHE COLLEGI, SPIEGA:
- TIPO DI RELAZIONE: causa-effetto? evoluzione? contemporaneità? successione?
- CATENA CAUSALE: perché questo evento ha portato a quell'altro?
- EVIDENZA: quale elemento nel documento supporta questa connessione temporale?

ESEMPIO:
"1585 → 1610: EVOLUZIONE CONTINUA. Il manierismo di fine Cinquecento (1585) evolve nel primo barocco perché [ragione]. L'evidenza nel documento [citazione] conferma questa transizione."

Ordina cronologicamente e mostra le relazioni con spiegazioni dettagliate.""",

    "vision": """Sei un esperto di iconografia e analisi visiva dell'arte.

ANALIZZA L'IMMAGINE OSSERVANDO:
1. Composizione e struttura spaziale
2. Soggetto principale e secondario
3. Palette cromatica e uso della luce
4. Tecnica pittorica/scultorea
5. Simbolismo e significato
6. Firma, data, iscrizioni
7. Stato di conservazione
8. Comparazione con opere coeve

PER OGNI ELEMENTO CHE OSSERVI, SPIEGA:
- COSA VEDO: descrizione oggettiva dell'elemento
- PERCHÉ È IMPORTANTE: significato iconografico o stilistico
- COME SI COLLEGA: relazione con altri elementi dell'opera o con altri artisti/movimenti

SPIEGA IL TUO RAGIONAMENTO per ogni elemento osservato.
Non fare affermazioni non supportate da ciò che vedi nell'immagine."""
}

# ============================================================
# FILE STORAGE
# ============================================================

def get_temp_dir():
    """Get temp directory for file storage"""
    server_dir = os.path.dirname(os.path.abspath(__file__))
    temp_dir = os.path.join(server_dir, "temp_uploads")
    os.makedirs(temp_dir, exist_ok=True)
    return temp_dir

# ============================================================
# API ROUTES
# ============================================================

@app.get("/")
async def root():
    """Serve main page"""
    return FileResponse("templates/index.html")

@app.get("/api/health")
async def health():
    """Health check"""
    return {"status": "ok", "version": "7.0"}

@app.post("/api/upload")
@limiter.limit("10/minute")
async def upload_file(request: Request, file: UploadFile = File(...)):
    """Upload and process a document"""
    temp_dir = get_temp_dir()
    file_id = str(uuid.uuid4())
    filename = file.filename or "unknown"
    file_path = os.path.join(temp_dir, f"{file_id}_{filename}")
    
    try:
        contents = await file.read()
        with open(file_path, 'wb') as f:
            f.write(contents)
        
        # Process based on file type
        ext = os.path.splitext(filename)[1].lower() if filename else ""
        
        if ext == '.pdf':
            result = doc_processor.process_pdf(file_path)
        elif ext in ['.png', '.jpg', '.jpeg', '.gif', '.webp']:
            result = doc_processor.process_image(file_path)
        else:
            result = doc_processor.process_text(file_path)
        
        # Extract knowledge if text
        if result.get("success") and result.get("type") in ["pdf", "text"]:
            knowledge = knowledge_extractor.extract(result.get("full_text", ""))
            result["knowledge"] = knowledge
        
        result["file_id"] = file_id
        result["filename"] = filename
        
        return result
        
    except Exception as e:
        return {"success": False, "error": str(e)}

@app.post("/api/analyze")
@limiter.limit("5/minute")
async def analyze_document(request: Request):
    """Analyze uploaded document with AI - Deep Reasoning Version"""
    try:
        body = await request.json()
        file_id = body.get("file_id")
        question = body.get("question", "Analizza questo documento in dettaglio, mostrando il tuo ragionamento")
        analysis_type = body.get("type", "deep")
        
        if not file_id:
            return {"success": False, "error": "No file_id provided"}
        
        # Find file
        temp_dir = get_temp_dir()
        matches = glob.glob(os.path.join(temp_dir, f"{file_id}_*"))
        
        if not matches:
            return {"success": False, "error": "File not found"}
        
        file_path = matches[0]
        filename = os.path.basename(file_path).split('_', 1)[-1] if '_' in os.path.basename(file_path) else os.path.basename(file_path)
        ext = os.path.splitext(filename)[1].lower() if filename else ""
        
        # Process file
        if ext == '.pdf':
            doc_data = doc_processor.process_pdf(file_path)
        elif ext in ['.png', '.jpg', '.jpeg', '.gif', '.webp']:
            doc_data = doc_processor.process_image(file_path)
        else:
            doc_data = doc_processor.process_text(file_path)
        
        if not doc_data.get("success"):
            return doc_data
        
        # Get images if available (PDF images OR direct image upload)
        images = doc_data.get("images", [])
        
        # Fix: se è un'immagine caricata direttamente, estrai image_data
        if doc_data.get("type") == "image" and doc_data.get("image_data"):
            images = [{
                "data": doc_data["image_data"],
                "name": doc_data.get("filename", "uploaded_image"),
                "source": "direct_upload"
            }]
        
        # Extract knowledge
        if doc_data.get("type") in ["pdf", "text"]:
            knowledge = knowledge_extractor.extract(doc_data.get("full_text", ""))
        else:
            knowledge = {}
        
        # Build enhanced analysis prompt
        context = doc_data.get("full_text", "")[:50000] if doc_data.get("type") in ["pdf", "text"] else ""
        num_pages = doc_data.get("num_pages", 1)
        
        if analysis_type == "compare":
            system_prompt = SYSTEM_PROMPTS["comparator"]
            prompt = f"""CONFERMA DI AVER ANALIZZATO INTEGRALMENTE il documento "{filename}" ({num_pages} pagine).

ISTRUZIONI PER IL RAGIONAMENTO:
- Prima di confrontare, analizza ogni elemento separatamente
- Mostra il tuo ragionamento per ogni connessione che estableisci
- Spiega perché hai identificato somiglianze e differenze

DATI ESTRATTI DAL DOCUMENTO:
- Pagine: {num_pages}
- Anni menzionati: {', '.join(knowledge.get('years', [])[:15])}
- Movimenti: {', '.join(knowledge.get('movements', []))}
- Concetti chiave: {', '.join(knowledge.get('key_concepts', [])[:15])}

CONTENUTO INTEGRALE DEL DOCUMENTO:
{context}

IMMAGINI NEL DOCUMENTO: {len(images)} immagini trovate

DOMANDA: {question}

ANALISI RICHIESTA:
1. Analizza il documento integralmente
2. SPIEGA IL TUO RAGIONAMENTO per ogni punto
3. Crea collegamenti logici
4. Fornisci citazioni con riferimenti [p. XX]"""

        elif analysis_type == "timeline":
            system_prompt = SYSTEM_PROMPTS["timeline_builder"]
            prompt = f"""CONFERMA DI AVER ANALIZZATO INTEGRALMENTE il documento "{filename}".

ISTRUZIONI PER IL RAGIONAMENTO:
- Estrai tutte le date in modo sistematico
- Spiega come hai stabilito la cronologia
- Mostra le connessioni causali tra gli eventi

DATI ESTRATTI:
- Anni menzionati: {knowledge.get('years', [])}
- Movimenti trovati: {', '.join(knowledge.get('movements', []))}

CONTENUTO DEL DOCUMENTO:
{context}

DOMANDA: {question}

Crea una timeline dettagliata SPIEGANDO il tuo ragionamento sulla cronologia."""
        
        elif analysis_type == "vision" or doc_data.get("type") == "image":
            system_prompt = SYSTEM_PROMPTS["vision"]
            img_info = f"Immagine: {filename}"
            if doc_data.get("dimensions"):
                img_info += f" - Dimensioni: {doc_data.get('dimensions')}"
            prompt = f"""{img_info}

ISTRUZIONI PER IL RAGIONAMENTO:
- Descrivi ogni elemento che osservi
- Spiega il significato di ciò che vedi
- Collega agli artisti/movimenti quando possibile

DOMANDA: {question}

Analizza l'immagine SPIEGANDO il tuo ragionamento per ogni elemento osservato."""
        
        else:  # deep analysis
            system_prompt = SYSTEM_PROMPTS["art_historian"]
            img_note = f"\nIMMAGINI NEL DOCUMENTO: {len(images)} immagini estratte" if images else ""
            prompt = f"""CONFERMA SEMPRE A INIZIO RISPOSTA DI AVER LETTO/VISTO L'INTERO MATERIALE.

**LIVELLO DI DETTAGLIO RICHIESTO - PER PRESENTAZIONI E SCRITTI ACCADEMICI**:
La tua risposta deve essere ESTREMAMENTE DETTAGLIATA e completa. Ogni argomento deve essere trattato con la profondità di un articolo accademico.

STRUTTURA OBBLIGATORIA PER OGNI CONNESSIONE:
🔗 CONNESSIONE: [Elemento A] ↔ [Elemento B]
   TIPO: [influenza diretta / derivazione / evoluzione / opposizione / contemporaneità / successione]
   PERCHÉ: [spiegazione dettagliata - minimo 2-3 frasi che spieghino il contesto storico e le circostanze]
   COME SI MANIFESTA: [evidenza specifica nel documento con citazione letterale - es: "la tecnica del chiaroscuro in X richiama quella in Y come dimostra [p. XX]"]

ESEMPIO DI CONNESSIONE ESPLICATA E APPROFONDITA:
🔗 CONNESSIONE: Caravaggio ↔ Artemisia Gentileschi
   TIPO: Influenza diretta / Evoluzione
   PERCHÉ: Artemisia Gentileschi (1593-1654 ca.) fu allieva e collaboratrice di Caravaggio a Roma intorno al 1595-1600. Durante questo periodo, assorbì la tecnica del naturalismo caravaggesco, caratterizzato dall'uso del chiaroscuro drammatico e dalla rappresentazione di soggetti quotidiani in contesti realistici. Questa influenza è documentata sia dalle opere del periodo romano sia dalle lettere della pittrice.
   COME SI MANIFESTA: "Il naturalismo drammatico di Gentileschi nella 'Judith e la ancella' (1613-1614, Pitti Firenze) riprende il chiaroscuro caravaggesco con una luce quasi teatrale che illumina il momento della decapitazione di Oloferne, come evidenziato in [p. 15]."

PER OGNI ARTISTA TRATTATO, FORNISCI:
1. Nome completo e date (nascita-morte)
2. Biografia sintetica con formazione e influenza ricevuta
3. Periodo di attività principale e luoghi
4. Caratteristiche stilistiche distintive del suo lavoro
5. Opere principali con descrizione dettagliata (soggetto, tecnica, dimensioni, collocazione attuale)
6. Influenza esercitata su altri artisti e movimenti
7. Ricezione critica nel tempo

PER OGNI OPERA TRATTATA, FORNISCI:
1. Titolo completo e datazione
2. Committente e contesto di commissione
3. Descrizione iconografica dettagliata
4. Analisi stilistica (composizione, colori, luce, tecnica)
5. Simbolismo e significato
6. Storia critica e ricezione
7. Collocazione attuale e condizioni

PER OGNI MOVIMENTO TRATTATO, FORNISCI:
1. Definizione e caratteristiche distintive
2. Periodo storico e area geografica
3. Artisti principali e loro ruolo
4. Opere rappresentative
5. Relazione con movimenti precedenti e successivi

STRUMENTI A DISPOSIZIONE:
- Se hai bisogno di approfondire elementi nel contesto, puoi cercare informazioni web
- Indica sempre le fonti quando usi info esterne

DOCUMENTO: "{filename}"
- Pagine totali: {num_pages}
- Immagini estratte: {len(images)}{img_note}

DATI ESTRATTI:
- Anni menzionati: {', '.join(knowledge.get('years', [])[:20])}
- Movimenti artistici: {', '.join(knowledge.get('movements', []))}
- Artisti identificati: {', '.join(knowledge.get('artists', [])[:10])}
- Opere citate: {len(knowledge.get('artworks', []))}
- Concetti chiave: {', '.join(knowledge.get('key_concepts', [])[:20])}

CONTENUTO INTEGRALE DEL DOCUMENTO:
{context}

DOMANDA: {question}

STRUTTURA LA RISPOSTA COSÌ:
1. CONFERMA di aver analizzato tutto il materiale
2. INTRODUZIONE: Contesto storico-culturale del materiale analizzato (minimo 1-2 paragrafi)
3. ANALISI DETTAGLIATA per ogni artista/opera/movimento (minimo 2-3 paragrafi ciascuno)
4. Per OGNI collegamento che identifichi, usa il formato 🔗 CONNESSIONE completo
5. Confronta e crea connessioni ESPLICANDO il tipo, il perché e come si manifesta
6. BIBLIOGRAFIA E FONTI: Elenca le fonti citate
7. CONCLUSIONE: Sintesi interpretativa critica (minimo 1-2 paragrafi)
8. SUGGERIMENTI: Possibili approfondimenti e ricerche future"""
        
        # Query LLM with images if available
        result = llm.complete(prompt, system_prompt, images if images else None)
        
        return {
            "success": True,
            "analysis": result.get("text", "Analisi completata"),
            "knowledge": knowledge,
            "model": result.get("model", "unknown"),
            "latency": result.get("latency", 0),
            "filename": filename,
            "has_images": len(images) > 0,
            "num_images": len(images),
            "num_pages": num_pages
        }
        
    except Exception as e:
        return {"success": False, "error": str(e)}

@app.post("/api/search")
@limiter.limit("5/minute")
async def search_web(request: Request):
    """Search web for additional context"""
    try:
        body = await request.json()
        query = body.get("query", "").strip()
        
        if not query:
            return {"success": False, "error": "Empty query"}
        
        result = web_searcher.search(query)
        return result
        
    except Exception as e:
        return {"success": False, "error": str(e)}

@app.post("/api/chat")
@limiter.limit("10/minute")
async def chat(request: Request):
    """Chat with AI about documents or general questions - Enhanced Reasoning with Connections"""
    try:
        body = await request.json()
        message = body.get("message", "").strip()
        has_documents = body.get("has_documents", False)
        file_ids = body.get("file_ids", [])  # Frontend can pass specific file IDs
        
        if not message:
            return {"success": False, "error": "Empty message"}
        
        # Process uploaded files if any
        documents_context = ""
        images_for_vision = []
        
        if has_documents:
            temp_dir = get_temp_dir()
            
            # Get all files in temp directory (most recent first)
            all_files = glob.glob(os.path.join(temp_dir, "*"))
            all_files.sort(key=os.path.getmtime, reverse=True)
            
            # Process most recent files (limit to last 5)
            recent_files = all_files[:5]
            
            for file_path in recent_files:
                filename = os.path.basename(file_path)
                ext = os.path.splitext(filename)[1].lower()
                
                # Skip non-document files
                if ext in ['.png', '.jpg', '.jpeg', '.gif', '.webp', '.pdf', '.txt']:
                    try:
                        if ext in ['.png', '.jpg', '.jpeg', '.gif', '.webp']:
                            # Process as image for vision
                            doc_data = doc_processor.process_image(file_path)
                            if doc_data.get("success") and doc_data.get("image_data"):
                                images_for_vision.append({
                                    "data": doc_data["image_data"],
                                    "name": filename,
                                    "source": "direct_upload"
                                })
                        elif ext == '.pdf':
                            doc_data = doc_processor.process_pdf(file_path)
                            if doc_data.get("success"):
                                text = doc_data.get("full_text", "")[:30000]
                                documents_context += f"\n\n📄 DOCUMENTO: {filename}\nPagine: {doc_data.get('num_pages', 0)}\n\nCONTENUTO:\n{text}"
                        elif ext == '.txt':
                            doc_data = doc_processor.process_text(file_path)
                            if doc_data.get("success"):
                                documents_context += f"\n\n📄 DOCUMENTO: {filename}\n\nCONTENUTO:\n{doc_data.get('full_text', '')[:30000]}"
                    except Exception as e:
                        logger.warning(f"Error processing file {filename}: {e}")
            
            logger.info(f"Chat with docs: {len(images_for_vision)} images, {len(documents_context)} chars text")
        
        # Enhanced prompt for document analysis with reasoning and connections
        if has_documents:
            system_prompt = SYSTEM_PROMPTS["art_historian"]
            
            # Add document context if available
            context_section = f"\n\n{documents_context}" if documents_context else ""
            image_section = f"\n\n🎨 IMMAGINI CARICATE: {len(images_for_vision)} immagini da analizzare" if images_for_vision else ""
            
            prompt = f"""{context_section}{image_section}

CONFERMA DI AVER ANALIZZATO INTEGRALMENTE tutti i documenti e immagini forniti.

DOMANDA: {message}

📚 ISTRUZIONI PER ANALISI CON RAGIONAMENTO E CONNESSIONI:

STRUTTURA OBBLIGATORIA PER OGNI CONNESSIONE:
🔗 CONNESSIONE: [Elemento A] ↔ [Elemento B]
   TIPO: [influenza diretta / derivazione / evoluzione / opposizione / contemporaneità / successione]
   PERCHÉ: [spiegazione del perché esiste questo collegamento]
   COME SI MANIFESTA: [evidenza specifica - cita sempre la fonte con riferimento]

ESEMPIO:
🔗 CONNESSIONE: Raffaello ↔ Michelangelo
   TIPO: Influenza diretta / Evoluzione
   PERCHÉ: Raffaello studiò le opere di Michelangelo a Firenze e Roma
   COME SI MANIFESTA: "Le figure michelangiolesche nelle Stanze Vaticane (1509-1511) mostrano l'influenza della Cappella Sistina come indicato in [doc1, p. 45]"

- SPIEGA il tuo ragionamento: "Osservo che...", "Questo indica che...", "Confronto con..."
- Crea collegamenti tra artisti, movimenti, periodi ESPLICANDO il tipo e il perché
- Confronta elementi comuni e differenze
- Citazioni precise con [nome file | p. XX]
- Se hai bisogno di approfondire, puoi cercare informazioni web

Struttura la risposta in modo chiaro usando il formato 🔗 per ogni connessione."""
        else:
            system_prompt = SYSTEM_PROMPTS["art_historian"]
            prompt = f"""{message}

SPIEGA IL TUO RAGIONAMENTO per ogni punto.
Per ogni collegamento che identifichi, specifica:
🔗 CONNESSIONE: [A] ↔ [B]
   TIPO: [tipo di collegamento]
   PERCHÉ: [perché esiste]
   COME: [come si manifesta]

Se parli di artisti, movimenti o opere, fornisci contesto storico-artistico approfondito.
Se hai bisogno di approfondire, puoi cercare informazioni web e indicare le fonti."""
        
        # Pass images to LLM if available (for vision analysis)
        images_to_pass = images_for_vision if images_for_vision else None
        result = llm.complete(prompt, system_prompt, images_to_pass)
        
        if result.get("success"):
            return {
                "success": True,
                "response": result["text"],
                "backend": result.get("model", "unknown"),
                "latency": f"{result.get('latency', 0):.2f}s"
            }
        else:
            return {
                "success": False,
                "error": result.get("error", "Unknown error")
            }
            
    except Exception as e:
        return {"success": False, "error": str(e)}

# ============================================================
# START SERVER
# ============================================================

if __name__ == "__main__":
    import uvicorn
    print("🤓 AgenteIA - Artificial Scholar v7.0")
    print("🚀 Server starting on http://localhost:8002")
    print("✨ Enhanced Reasoning & Vision Analysis")
    uvicorn.run(app, host="127.0.0.1", port=8002, log_level="info")
