#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ADVANCED AGENTIC REASONING LLM SYSTEM v5 (ENTERPRISE)
Production-ready 8-tier multi-backend system with dynamic modal prompts
AGENTIC MODE: Truth-Only, Document Analysis, Multi-Modal, ABA Bari Focused

TARGET USERS: ABA Bari Students/Faculty (Fashion Design, Arts, Design)

CAPABILITIES:
  - Dynamic system prompts per modalita (Ragionamento, Analisi, Fashion, Esami)
  - Document analysis (PDF, PPTX, DOCX, immagini)
  - Vision capabilities (image recognition, OCR)
  - Truth-only responses (no speculation)
  - Italian-only output
  - Workflow creation & task planning

8-TIER VERIFIED FALLBACK:
  1. HuggingFace (0.83s - FASTEST)
  2. Groq llama-3.3-70b (1.69s - ULTRA-FAST)
  3. Mistral API (7.78s - RELIABLE)
  4. DeepSeek Chat (2-3s - POWERFUL)
  5. Google Gemini 2.5 (14.31s - HIGH QUALITY)
  6. OpenRouter free (15.45s - FREE TIER)
  7. Ollama Llama2 local (49.97s - OFFLINE)
  8. Ollama Cloud (variable - SCALABLE)
"""

from typing import Optional, Tuple, Dict, List, Callable
from dataclasses import dataclass
from llama_index.core.llms import LLM, ChatMessage, ChatResponse, CompletionResponse, LLMMetadata
from pydantic import ConfigDict
import os
import json
import logging
import time
import asyncio
from typing import ClassVar

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CognitiveStreamEmitter:
    """
    Emette eventi di pensiero reale in streaming verso il client WebSocket.
    NON simula - descrive ogni decisione effettiva mentre accade.
    """
    def __init__(self, websocket):
        self.ws = websocket
        self.start_time = time.time()

    async def emit(self, phase: str, content: str, model: str = None):
        elapsed = round(time.time() - self.start_time, 2)
        model_tag = f"[{model}] " if model else ""
        await self.ws.send_json({
            "type": "thinking_stream",
            "phase": phase,
            "content": f"{model_tag}{content}",
            "elapsed_s": elapsed
        })
        await asyncio.sleep(0)


@dataclass
class ResponseQuality:
    word_count: int
    depth_score: float
    answer_coverage: float
    confidence: float
    needs_escalation: bool
    reason: str


class ResponseQualityEvaluator:
    """
    Valuta se una risposta e abbastanza profonda per lo studente
    o se richiede escalation a un modello piu capace.
    """
    DEPTH_KEYWORDS = [
        "perche","quindi","tuttavia","d'altra parte","in contrasto",
        "analogamente","in particolare","ad esempio","conseguentemente",
        "analizzando","confrontando","storicamente","artisticamente",
        "influenza","stile","periodo","corrente","movimento","opera",
        "confronto","critica","interpretazione","significato","contesto"
    ]

    def evaluate(self, response: str, question: str) -> dict:
        words = response.split()
        word_count = len(words)
        depth_hits = sum(1 for w in words if w.lower().strip(".,;:") in self.DEPTH_KEYWORDS)

        # Heuristic coverage: if question has analysis verbs, require longer answers
        complex_triggers = ["confronta", "analizza", "spiega", "collega", "saggio", "presentazione"]
        is_complex = any(t in question.lower() for t in complex_triggers)

        depth_score = round(min(depth_hits / 5.0, 1.0), 2)
        answer_coverage = round(min(word_count / 400.0, 1.0), 2) if is_complex else round(min(word_count / 200.0, 1.0), 2)

        needs_escalation = (
            depth_score < 0.5 or
            (is_complex and word_count < 180) or
            (not is_complex and word_count < 80)
        )

        reason = "risposta adeguata"
        if needs_escalation:
            if depth_score < 0.5:
                reason = f"profondita analitica insufficiente ({depth_hits} indicatori)"
            elif is_complex and word_count < 180:
                reason = f"risposta breve per domanda complessa ({word_count} parole)"
            else:
                reason = f"risposta troppo breve ({word_count} parole)"

        return ResponseQuality(
            word_count=word_count,
            depth_score=depth_score,
            answer_coverage=answer_coverage,
            confidence=round((depth_score + answer_coverage) / 2, 2),
            needs_escalation=needs_escalation,
            reason=reason
        )


async def escalate_response(
    original_question: str,
    previous_response: str,
    previous_model: str,
    file_context: str,
    query_function,
    emitter: CognitiveStreamEmitter = None
) -> str:
    """
    Passa la risposta al modello successivo con contesto completo.
    Il modello successivo NON riparte da zero - integra e approfondisce.
    """
    if emitter:
        await emitter.emit(
            "escalation_start",
            f"Risposta di {previous_model} valutata come superficiale. "
            f"Costruisco prompt di escalation con contesto completo...",
            model=previous_model
        )
        await emitter.emit(
            "escalation_prompt",
            "Invio a modello superiore: domanda originale + risposta base + "
            "istruzione di integrare (non ripetere)..."
        )

    escalation_prompt = (
        f"Hai ricevuto questa domanda da uno studente:\n{original_question}\n\n"
        f"Il modello {previous_model} ha gia risposto cosi:\n"
        f"---\n{previous_response}\n---\n\n"
        f"Il tuo compito NON e rispondere da zero.\n"
        f"Il tuo compito e:\n"
        f"1. Identificare cosa manca o e superficiale nella risposta precedente\n"
        f"2. Approfondire quegli aspetti specifici con analisi critica\n"
        f"3. Aggiungere connessioni, riferimenti, esempi, confronti artistici\n"
        f"4. Se presente il documento, citare pagine e sezioni precise\n"
        f"5. Struttura output: [Integrazione] poi [Approfondimento] poi [Connessioni]\n"
        + (f"\nContesto documento:\n{file_context}" if file_context else "")
    )

    loop = asyncio.get_event_loop()
    text, success, metadata = await loop.run_in_executor(
        None,
        lambda: query_function(escalation_prompt)
    )

    if emitter:
        await emitter.emit(
            "escalation_complete",
            "Risposta arricchita ricevuta. Invio al client.",
        )

    return text


class AdvancedReasoningLLM(LLM):
    """
    Production LLM system with 8-tier fallback and Italian-only responses
    """
    
    model_config = ConfigDict(arbitrary_types_allowed=True, extra='allow')
    
    context_window: int = 8000
    num_output: int = 512
    attempts_log: List[str] = []  # Log of backend attempts for this completion
    on_attempt: Optional[callable] = None  # Callback when attempting a backend (sync function)
    temperature: float = 0.3
    active_backend: Optional[str] = None
    active_model: Optional[str] = None
    last_response_metadata: Dict = {}
    stream_callback: Optional[Callable[[str, str, Optional[str], Optional[Dict]], None]] = None
    
    # Common English words for language detection
    COMMON_ENGLISH_WORDS: ClassVar[set] = {
        'hello', 'hi', 'hey', 'thank', 'thanks', 'yes', 'no', 'ok', 'okay', 
        'the', 'and', 'of', 'to', 'in', 'is', 'a', 'for', 'that', 'with',
        'why', 'what', 'how', 'who', 'when', 'where', 'because', 'if', 'would',
        'could', 'should', 'can', 'do', 'does', 'did', 'have', 'has', 'had',
        'be', 'are', 'as', 'from', 'by', 'on', 'at', 'this', 'it', 'or',
        'but', 'not', 'just', 'only', 'some', 'my', 'your', 'his', 'her',
        'good', 'bad', 'new', 'old', 'big', 'small', 'red', 'blue', 'green'
    }
    
    def _detect_language_ratio(self, text: str) -> float:
        """Detect ratio of English words in text (0.0 = Italian, 1.0 = English)"""
        words = text.lower().split()
        if not words:
            return 0.0
        
        english_count = sum(1 for w in words if w.strip('.,!?;:') in self.COMMON_ENGLISH_WORDS)
        return english_count / len(words)
    
    def _force_italian_response(self, text: str) -> str:
        """Post-process response to enforce Italian-only output (gentle filtering)"""
        # Only apply harsh filtering if VERY English-heavy (> 60% English words)
        english_ratio = self._detect_language_ratio(text)
        
        if english_ratio > 0.60:
            # Extract mostly-Italian parts
            import re
            sentences = re.split(r'[.!?]\s+', text)
            italian_sentences = []
            
            for sent in sentences:
                sent_ratio = self._detect_language_ratio(sent)
                if sent_ratio < 0.50:  # Keep mostly-Italian sentences
                    italian_sentences.append(sent.strip())
            
            if italian_sentences and len(italian_sentences) >= 2:  # At least 2 sentences
                filtered_text = '. '.join(italian_sentences)
                if filtered_text:
                    return filtered_text + '.'
        
        return text

    def _emit_stream(self, phase: str, content: str, model: Optional[str] = None, data: Optional[Dict] = None):
        """Send streaming event to external callback (thread-safe via queue)."""
        if self.stream_callback:
            try:
                self.stream_callback(phase, content, model, data or {})
            except Exception as e:
                logger.debug(f"[STREAM] Callback error: {e}")
    
    def _force_italian_aggressive(self, text: str) -> str:
        """BRUTAL Italian enforcement for Ollama - filters out English sentences completely"""
        import re
        
        english_ratio = self._detect_language_ratio(text)
        logger.info(f"[FILTER] English ratio: {english_ratio:.2%} | Text length: {len(text)} chars")
        logger.info(f"[FILTER] Input (first 100 chars): {text[:100]}...")
        
        # If mostly Italian already, do gentle word replacement only
        if english_ratio <= 0.40:
            logger.info(f"[FILTER] Mode: GENTLE (ratio <= 40%)")
            # Just clean up obvious English words
            replacements = {
                r'\bspecifically\s+designed\s+for': 'specificamente progettato per',
                r'\bdesigned\s+for': 'progettato per',
                r'\bHello\b': 'Ciao',
                r'\bhi\b': 'ciao',
                r'\bthanks\b': 'grazie',
                r'\bpleasure\b': 'piacere',
                r'\bHow are you\b': 'Come stai',
            }
            result = text
            for pattern, replacement in replacements.items():
                result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)
            logger.info(f"[FILTER] Output (first 100 chars): {result[:100]}...")
            return result
        
        # If heavily English (> 40%), FILTER SENTENCES AGGRESSIVELY
        logger.info(f"[FILTER] Mode: AGGRESSIVE (ratio > 40%) - Splitting into sentences...")
        sentences = re.split(r'[.!?]\s+', text)
        logger.info(f"[FILTER] Found {len(sentences)} sentences")
        
        italian_sentences = []
        
        for i, sent in enumerate(sentences):
            if not sent.strip():
                continue
            
            sent_ratio = self._detect_language_ratio(sent)
            keep_sentence = sent_ratio < 0.30
            
            logger.info(f"[FILTER] Sentence {i}: ratio={sent_ratio:.2%}, keep={keep_sentence}")
            logger.info(f"[FILTER]   Text: {sent[:60]}...")
            
            # ONLY keep sentences that are < 30% English
            if keep_sentence:
                italian_sentences.append(sent.strip())
        
        # If we filtered everything, keep at least first sentence
        if not italian_sentences and sentences:
            logger.warning("[FILTER] All sentences filtered! Keeping first sentence as fallback")
            italian_sentences = [sentences[0].strip()]
        
        if italian_sentences:
            result = '. '.join(italian_sentences)
            if result and not result.endswith('.'):
                result += '.'
            logger.info(f"[FILTER] AGGRESSIVE filtering result ({len(italian_sentences)} sentences kept)")
            logger.info(f"[FILTER] Output (first 100 chars): {result[:100]}...")
            return result
        
        # Fallback to original if filtering removes everything
        logger.error("[FILTER] FALLBACK: returning original text (filtering failed completely)")
        return text
    
    def __init__(self, temperature: float = 0.3, max_tokens: int = 512):
        super().__init__()
        self.temperature = temperature
        self.num_output = max_tokens
        self.active_backend = None
        self.active_model = None
        self.last_response_metadata = {}
        self.attempts_log = []  # Log of attempts for current completion
        self.stream_callback = None
        logger.info("✅ AdvancedReasoningLLM v5 initialized (Agentic, Truth-Only)")
    
    def _get_system_prompt(self, modalita: str = "generale") -> str:
        """Get dynamic system prompt based on selected modalità"""
        
        # Base prompt TRUTH-ONLY
        base = """# IDENTITÀ E MISSIONE

Sei un assistente AI personale ultra-preciso, proattivo e affidabile. Il tuo utente è uno studente/docente presso l'Accademia di Belle Arti di Bari (ABA Bari), con focus su Fashion Design, arti visive e design.

## REGOLA ASSOLUTA: SOLO VERITÀ
- Non inventare MAI dati, date, nomi, citazioni, tendenze, normative o fatti.
- Se non sai qualcosa con certezza, dillo esplicitamente: "Non ho questa informazione con certezza."
- Distingui sempre tra: FATTO VERIFICATO / MIA STIMA / IPOTESI DA CONFERMARE.
- Non completare lacune con "probabilmente" senza segnalarlo chiaramente.

## LINGUAGGIO: ITALIANO ESCLUSIVAMENTE
- Rispondi sempre in italiano perfetto.
- Tecnico-accademico per contesto ABA/moda/arte.
- Pratico e diretto per vita quotidiana.
- Mai condiscendente, mai prolisso."""

        # Modalità specifiche
        if modalita == "ragionamento":
            return base + """

## MODALITÀ: RAGIONAMENTO LOGICO
- Fornisci step-by-step del processo di pensiero.
- Identifica assunzioni, premesse, conclusioni.
- Segnala quando servono chiarimenti dall'utente.
- Output: numeri ordinati con inferenze esplicite."""

        elif modalita == "analisi":
            return base + """

## MODALITÀ: ANALISI STRUTTURALE
- Analizza documenti, testi, immagini, dati forniti.
- Estrai punti chiave, criticità, lacune.
- Struttura: Osservazioni → Problemi → Opportunità.
- Sii specifico, citando sezioni/elementi concreti.
- Output: tabelle, schemi, mappe when relevant."""

        elif modalita == "fashion_design":
            return base + """

## MODALITÀ: FASHION DESIGN & MODA
Sei specializzato in Fashion Design, storia della moda, modellistica, textile design, trend forecasting.
- Analizza collezioni, storytelling moda, sostenibilità.
- Aiuta con illustrazione moda, comunicazione brand, portfolio.
- Conosci ABA Bari Fashion Design program (I e II livello).
- Output: brief progetti, analisi trend, consigli design."""

        elif modalita == "esami":
            return base + """

## MODALITÀ: PREPARAZIONE ESAMI
- Prepara schede di studio sintetiche.
- Genera domande simulate con risposte modello.
- Crea mappe concettuali per discipline ABA (Storia Arte, Estetica, Comunicazione Visiva, ecc.).
- Focus: memoria, connessioni, argomentazione critica.
- Output: chiaro, facile da ripassare, con reference alle fonti."""

        elif modalita == "presentazioni":
            return base + """

## MODALITÀ: CREAZIONE PRESENTAZIONI
- Aiuta con struttura narrativa di slide.
- Suggerisci layout visivo, gerarchia contenuto.
- Bilancia contenuto teorico + elementi visivi.
- Output: outline slide, suggerimenti design, note relatore."""

        elif modalita == "documenti":
            return base + """

## MODALITÀ: GESTIONE DOCUMENTI
- Analizza file caricati (PDF, PPTX, DOCX, immagini).
- Estrai punti chiave, riassumi, identifica lacune.
- Riconosci layout visivi (slide, tavole, moodboard).
- Output: rapporto strutturato, azioni suggerite."""

        else:  # "generale"
            return base + """

## MODALITÀ: ASSISTENZA GENERALE
- Rispondi a domande quotidiane, studio, organizzazione.
- Crea liste, promemoria, workflow.
- Adatta profondità al contesto dell'utente.
- Proponi sempre azione concreta al termine."""
    
    def _query_huggingface(self, prompt: str) -> Tuple[str, bool, Dict]:
        """HuggingFace Inference Providers - FASTEST (0.83s)"""
        try:
            from huggingface_hub import InferenceClient
            
            hf_token = os.getenv("HF_TOKEN")
            if not hf_token:
                return "", False, {}
            
            client = InferenceClient(api_key=hf_token)
            start = time.time()
            response = client.chat.completions.create(
                model="openai/gpt-oss-120b:fastest",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=self.num_output,
                temperature=self.temperature,
            )
            latency = time.time() - start
            
            text = response.choices[0].message.content
            metadata = {
                "model": "gpt-oss-120b:fastest",
                "provider": "HuggingFace",
                "latency": f"{latency:.2f}s",
                "tokens": len(text.split())
            }
            
            return text, True, metadata
        except Exception as e:
            logger.error(f"[HUGGINGFACE DETAILED ERROR]")
            logger.error(f"  Exception type: {type(e).__name__}")
            logger.error(f"  Full message: {str(e)}")
            logger.error(f"  API Key exists: {bool(hf_token)}")
            import traceback
            logger.error(f"  Traceback: {traceback.format_exc()}")
            return "", False, {}
    
    def _query_groq(self, prompt: str) -> Tuple[str, bool, Dict]:
        """Groq llama-3.3-70b - ULTRA-FAST (1.69s)"""
        try:
            from groq import Groq
            
            groq_key = os.getenv("GROQ_API_KEY")
            if not groq_key:
                logger.warning("[GROQ] API Key not set!")
                return "", False, {}
            
            logger.info(f"[GROQ] API Key exists, length: {len(groq_key)}")
            client = Groq(api_key=groq_key)
            logger.info(f"[GROQ] Client created successfully")
            
            start = time.time()
            logger.info(f"[GROQ] Sending request (prompt length: {len(prompt)} chars)")
            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                max_tokens=self.num_output,
            )
            latency = time.time() - start
            logger.info(f"[GROQ] Response received in {latency:.2f}s")
            
            logger.info(f"[GROQ] Response type: {type(response)}")
            logger.info(f"[GROQ] Choices available: {len(response.choices)}")
            
            if not response.choices:
                logger.error("[GROQ] No choices in response!")
                return "", False, {}
            
            text = response.choices[0].message.content
            logger.info(f"[GROQ] Text extracted, length: {len(text) if text else 0}")
            
            if not text:
                logger.error("[GROQ] Extracted text is empty!")
                return "", False, {}
            
            metadata = {
                "model": "llama-3.3-70b-versatile",
                "provider": "Groq",
                "latency": f"{latency:.2f}s",
                "tokens": len(text.split())
            }
            
            logger.info(f"[GROQ] SUCCESS! Returning text of length {len(text)}")
            return text, True, metadata
        except Exception as e:
            logger.error(f"[GROQ DETAILED ERROR]")
            logger.error(f"  Exception type: {type(e).__name__}")
            logger.error(f"  Full message: {str(e)}")
            logger.error(f"  API Key exists: {bool(groq_key)}")
            import traceback
            logger.error(f"  Traceback: {traceback.format_exc()}")
            return "", False, {}
    
    def _query_mistral(self, prompt: str) -> Tuple[str, bool, Dict]:
        """Mistral API - RELIABLE (7.78s)"""
        try:
            import httpx
            
            mistral_key = os.getenv("MISTRAL_API_KEY")
            if not mistral_key:
                return "", False, {}
            
            url = "https://api.mistral.ai/v1/chat/completions"
            headers = {
                "Authorization": f"Bearer {mistral_key}",
                "Content-Type": "application/json"
            }
            payload = {
                "model": "mistral-small-latest",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": self.temperature,
                "max_tokens": self.num_output
            }
            
            start = time.time()
            with httpx.Client(timeout=30.0) as client:
                response = client.post(url, json=payload, headers=headers)
                response.raise_for_status()
            latency = time.time() - start
            
            data = response.json()
            text = data["choices"][0]["message"]["content"]
            metadata = {
                "model": "mistral-small-latest",
                "provider": "Mistral",
                "latency": f"{latency:.2f}s",
                "tokens": len(text.split())
            }
            
            return text, True, metadata
        except Exception as e:
            logger.error(f"[MISTRAL DETAILED ERROR]")
            logger.error(f"  Exception type: {type(e).__name__}")
            logger.error(f"  Full message: {str(e)}")
            logger.error(f"  API Key exists: {bool(mistral_key)}")
            import traceback
            logger.error(f"  Traceback: {traceback.format_exc()}")
            return "", False, {}
    
    def _query_gemini(self, prompt: str) -> Tuple[str, bool, Dict]:
        """Google Gemini 2.5 Flash - HIGH QUALITY, STABLE (1-2s)"""
        try:
            import google.generativeai as genai
            
            gemini_key = os.getenv("GOOGLE_API_KEY")
            if not gemini_key:
                return "", False, {}
            
            genai.configure(api_key=gemini_key)
            model = genai.GenerativeModel("gemini-2.5-flash")
            
            start = time.time()
            response = model.generate_content(prompt)
            latency = time.time() - start
            
            text = response.text
            metadata = {
                "model": "gemini-2.5-flash",
                "provider": "Google",
                "latency": f"{latency:.2f}s",
                "tokens": len(text.split())
            }
            
            return text, True, metadata
        except Exception as e:
            logger.error(f"[GEMINI DETAILED ERROR]")
            logger.error(f"  Exception type: {type(e).__name__}")
            logger.error(f"  Full message: {str(e)}")
            logger.error(f"  API Key exists: {bool(gemini_key)}")
            import traceback
            logger.error(f"  Traceback: {traceback.format_exc()}")
            return "", False, {}
    
    def _query_gemini_vision(self, prompt: str, image_path: str) -> Tuple[str, bool, Dict]:
        """Google Gemini 2.5 Flash with Vision - Image Analysis Capability"""
        try:
            import google.generativeai as genai
            from PIL import Image
            
            gemini_key = os.getenv("GOOGLE_API_KEY")
            if not gemini_key:
                return "", False, {}
            
            # Check if image file exists
            if not os.path.exists(image_path):
                return "", False, {}
            
            genai.configure(api_key=gemini_key)
            model = genai.GenerativeModel("gemini-2.5-flash")
            
            # Load image
            image = Image.open(image_path)
            
            # Extract image metadata (filename, size)
            file_size_kb = os.path.getsize(image_path) / 1024
            image_info = f"[Image: {os.path.basename(image_path)}, {image.width}x{image.height}px, {file_size_kb:.1f}KB]"
            
            # Build content with image and prompt
            start = time.time()
            response = model.generate_content([prompt, image])
            latency = time.time() - start
            
            text = response.text
            metadata = {
                "model": "gemini-2.5-flash-vision",
                "provider": "Google Vision",
                "latency": f"{latency:.2f}s",
                "tokens": len(text.split()),
                "image_info": image_info
            }
            
            logger.info(f"   Gemini Vision analyzed: {image_info}")
            return text, True, metadata
        except Exception as e:
            logger.warning(f"Gemini Vision error: {str(e)[:50]}")
            return "", False, {}
    
    def _query_openrouter(self, prompt: str) -> Tuple[str, bool, Dict]:
        """OpenRouter free models - FREE TIER (15.45s)"""
        try:
            import httpx
            
            openrouter_key = os.getenv("OPENROUTER_API_KEY")
            if not openrouter_key:
                return "", False, {}
            
            url = "https://openrouter.ai/api/v1/chat/completions"
            headers = {
                "Authorization": f"Bearer {openrouter_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "http://localhost",
                "X-Title": "AgenteIA",
            }
            payload = {
                "model": "openrouter/free",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": self.temperature,
                "max_tokens": self.num_output
            }
            
            start = time.time()
            with httpx.Client(timeout=30.0) as client:
                response = client.post(url, json=payload, headers=headers)
                response.raise_for_status()
            latency = time.time() - start
            
            data = response.json()
            text = data["choices"][0]["message"]["content"]
            metadata = {
                "model": "openrouter/free",
                "provider": "OpenRouter",
                "latency": f"{latency:.2f}s",
                "tokens": len(text.split())
            }
            
            return text, True, metadata
        except Exception as e:
            logger.warning(f"OpenRouter error: {str(e)[:50]}")
            return "", False, {}
    
    def _query_llama2(self, prompt: str) -> Tuple[str, bool, Dict]:
        """Ollama Llama2 local - OFFLINE, ZERO COST (49.97s)"""
        try:
            from ollama import Client
            
            client = Client(host='http://localhost:11434')
            
            # Extract system prompt and user message from combined prompt
            # Format is: "SYSTEM PROMPT\n\n---\n\nUTENTE: USER MESSAGE"
            parts = prompt.split("\n\n---\n\n")
            if len(parts) == 2:
                system_prompt = parts[0]
                user_message = parts[1].replace("UTENTE: ", "").strip()
            else:
                system_prompt = "Sei un assistente AI utile. Rispondi sempre in italiano."
                user_message = prompt
            
            start = time.time()
            
            # Use chat API for proper system prompt handling
            response = client.chat(
                model="llama2",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message}
                ],
                stream=False,
            )
            latency = time.time() - start
            
            text = response['message']['content']
            
            # AGGRESSIVE Italian enforcement for Ollama specifically
            text = self._force_italian_aggressive(text)
            
            metadata = {
                "model": "llama2",
                "provider": "Ollama (Local)",
                "latency": f"{latency:.2f}s",
                "tokens": response.get('eval_count', len(text.split()))
            }
            
            return text, True, metadata
        except Exception as e:
            logger.warning(f"Llama2 error: {str(e)[:50]}")
            return "", False, {}
    
    def _query_deepseek(self, prompt: str) -> Tuple[str, bool, Dict]:
        """DeepSeek Chat (OpenAI-compatible API) - POWERFUL NEW BACKEND (2-3s)"""
        try:
            from openai import OpenAI
            
            deepseek_key = os.getenv("DEEPSEEK_API_KEY")
            if not deepseek_key:
                return "", False, {}
            
            # DeepSeek uses OpenAI-compatible API
            client = OpenAI(api_key=deepseek_key, base_url="https://api.deepseek.com")
            
            start = time.time()
            response = client.chat.completions.create(
                model="deepseek-chat",
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                max_tokens=self.num_output,
            )
            latency = time.time() - start
            
            text = response.choices[0].message.content
            metadata = {
                "model": "deepseek-chat",
                "provider": "DeepSeek",
                "latency": f"{latency:.2f}s",
                "tokens": len(text.split())
            }
            
            logger.info(f"   ✅ DeepSeek responded in {latency:.2f}s")
            return text, True, metadata
        except Exception as e:
            logger.warning(f"DeepSeek error: {str(e)[:50]}")
            return "", False, {}
    
    def _query_ollama_cloud(self, prompt: str) -> Tuple[str, bool, Dict]:
        """Ollama Cloud API - REMOTE CLOUD ENDPOINT WITH API KEY SUPPORT"""
        try:
            import requests
            from ollama import Client
            
            # Get custom Ollama endpoint from env
            ollama_endpoint = os.getenv("OLLAMA_CLOUD_ENDPOINT", "http://localhost:11434")
            ollama_model = os.getenv("OLLAMA_CLOUD_MODEL", "ministral-3:8b")
            ollama_api_key = os.getenv("OLLAMA_CLOUD_API_KEY", "")
            
            # Skip if using default localhost (not configured for cloud)
            if ollama_endpoint == "http://localhost:11434" and not ollama_api_key:
                return "", False, {}
            
            # Check if this is a remote endpoint (not localhost)
            is_remote = "localhost" not in ollama_endpoint and "127.0.0.1" not in ollama_endpoint
            
            if is_remote and ollama_api_key:
                # Use requests for remote API with authentication
                logger.info(f"   [OLLAMA CLOUD] Using remote endpoint with API key authentication")
                
                headers = {"Authorization": f"Bearer {ollama_api_key}"}
                payload = {
                    "model": ollama_model,
                    "prompt": prompt,
                    "stream": False  # We want full response, not streaming
                }
                
                start = time.time()
                response = requests.post(
                    f"{ollama_endpoint}/generate",
                    json=payload,
                    headers=headers,
                    timeout=60,  # Cloud can be slower
                    stream=False
                )
                latency = time.time() - start
                
                response.raise_for_status()
                
                # Handle both streaming and non-streaming responses
                full_response = ""
                
                # If response is JSON lines (streaming), concatenate all
                if response.headers.get('content-type', '').startswith('application/x-ndjson') or '\n' in response.text:
                    import json
                    for line in response.text.strip().split('\n'):
                        if line:
                            data = json.loads(line)
                            full_response += data.get('response', '')
                            if data.get('done', False):
                                break
                else:
                    # Single JSON response
                    data = response.json()
                    full_response = data.get('response', '')
                
                text = full_response
                
            else:
                # Use ollama Client for localhost
                logger.info(f"   [OLLAMA] Using local/direct endpoint")
                client = Client(host=ollama_endpoint)
                
                start = time.time()
                response = client.generate(
                    model=ollama_model,
                    prompt=prompt,
                    stream=False,
                )
                latency = time.time() - start
                text = response['response']
            
            metadata = {
                "model": ollama_model,
                "provider": "Ollama (Cloud)" if is_remote else "Ollama (Local)",
                "endpoint": ollama_endpoint,
                "latency": f"{latency:.2f}s",
                "tokens": len(text.split())
            }
            
            logger.info(f"   ✅ Ollama Cloud responded in {latency:.2f}s")
            return text, True, metadata
        except Exception as e:
            logger.error(f"[OLLAMA CLOUD DETAILED ERROR]")
            logger.error(f"  Exception type: {type(e).__name__}")
            logger.error(f"  Full message: {str(e)}")
            logger.error(f"  Endpoint: {ollama_endpoint}")
            logger.error(f"  Model: {ollama_model}")
            logger.error(f"  Has API Key: {bool(ollama_api_key)}")
            import traceback
            logger.error(f"  Traceback: {traceback.format_exc()}")
            return "", False, {}
    
    def _query_kimi(self, prompt: str) -> Tuple[str, bool, Dict]:
        """
        Kimi K2.5 (Moonshot AI) - ULTRA-REASONING MODEL
        Excels at: Math, Logic, Philosophy, Complex Reasoning
        Speed: ~2-3s | Quality: ⭐⭐⭐⭐⭐ | Free Tier: 10M tokens/month
        """
        try:
            import requests
            
            kimi_endpoint = os.getenv("KIMI_ENDPOINT", "https://api.moonshot.ai/v1")
            kimi_model = os.getenv("KIMI_MODEL", "moonshot-v1-32k")
            kimi_key = os.getenv("KIMI_API_KEY")
            
            if not kimi_key:
                return "", False, {}
            
            headers = {
                "Authorization": f"Bearer {kimi_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": kimi_model,
                "messages": [
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.3,
                "max_tokens": 1024
            }
            
            start = time.time()
            response = requests.post(
                f"{kimi_endpoint}/chat/completions",
                json=payload,
                headers=headers,
                timeout=60
            )
            latency = time.time() - start
            
            if response.status_code == 200:
                data = response.json()
                text = data["choices"][0]["message"]["content"]
                
                metadata = {
                    "model": kimi_model,
                    "provider": "Moonshot AI (Kimi K2.5)",
                    "latency": f"{latency:.2f}s",
                    "tokens": len(text.split()),
                    "reasoning_capable": True
                }
                
                logger.info(f"   ✅ Kimi K2.5 responded in {latency:.2f}s")
                return text, True, metadata
            else:
                logger.warning(f"   Kimi API error {response.status_code}: {response.text[:200]}")
                return "", False, {}
                
        except Exception as e:
            logger.error(f"[KIMI K2.5 ERROR] {type(e).__name__}: {str(e)}")
            return "", False, {}
    
    def _query_glm5(self, prompt: str) -> Tuple[str, bool, Dict]:
        """
        GLM-5 (Zhipu AI) via OpenRouter - TOP-TIER REASONING
        Excels at: Coding, Logic, Math, 200k context window
        Speed: ~3-4s | Quality: ⭐⭐⭐⭐⭐ | Free Tier: 90k req/day
        Uses OpenRouter aggregator API
        """
        try:
            import requests
            
            openrouter_key = os.getenv("OPENROUTER_API_KEY")
            if not openrouter_key:
                return "", False, {}
            
            headers = {
                "Authorization": f"Bearer {openrouter_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://agente-ia.local",
                "X-Title": "AgentIA-Advanced-Reasoning"
            }
            
            payload = {
                "model": "zhipu/glm-5",
                "messages": [
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.3,
                "max_tokens": 1024
            }
            
            start = time.time()
            response = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                json=payload,
                headers=headers,
                timeout=60
            )
            latency = time.time() - start
            
            if response.status_code == 200:
                data = response.json()
                text = data["choices"][0]["message"]["content"]
                
                metadata = {
                    "model": "glm-5",
                    "provider": "Zhipu AI (via OpenRouter)",
                    "latency": f"{latency:.2f}s",
                    "tokens": len(text.split()),
                    "context_window": "200k tokens",
                    "reasoning_capable": True
                }
                
                logger.info(f"   ✅ GLM-5 responded in {latency:.2f}s")
                return text, True, metadata
            else:
                logger.warning(f"   GLM-5 API error {response.status_code}")
                return "", False, {}
                
        except Exception as e:
            logger.error(f"[GLM-5 ERROR] {type(e).__name__}: {str(e)}")
            return "", False, {}
    
    def _get_backend_chain(self) -> List[Tuple[str, callable]]:
        """5-tier OPTIMIZED fallback chain - NO Google Gemini (free tier quota: 20/day)
        
        ✅ ACTIVE BACKENDS (Automatic Fallback Chain):
        1. Groq llama-3.3-70b           0.71s  ⚡⚡⚡ ULTRAVELOCE
        2. Mistral API                  0.87s  ⚡⚡  VELOCE
        3. HuggingFace gpt-oss-120b     1.03s  ⚡⚡  VELOCE
        4. Ollama Cloud (if configured) 8.35s  💾 CLOUD FALLBACK
        5. Ollama Llama2 (local)       16.10s  🖥️  OFFLINE FALLBACK
        
        ⚠️ AVAILABLE FOR MANUAL SELECTION (not in fallback chain):
        - Google Gemini 2.5 Flash (FREE TIER: only 20 req/day - unreliable)
        
        ❌ REMOVED (not functional):
        - DeepSeek Chat (402: insufficient balance)
        - OpenRouter free (404: endpoint not found)
        
        FALLBACK STRATEGY:
        - Best case (Groq works): 0.71s ✅
        - Degraded (Groq fails): 0.87s (Mistral)
        - Major fallback: 1.03s (HF)
        - Cloud needed: 8.35s (Ollama Cloud)
        - Offline mode: 16.10s (Ollama Local)
        """
        return [
            ("Groq llama-3.3-70b", self._query_groq),               # 0.71s FASTEST
            ("Mistral API", self._query_mistral),                   # 0.87s FAST
            ("HuggingFace Inference", self._query_huggingface),     # 1.03s FAST
            ("Ollama Cloud", self._query_ollama_cloud),             # 8.35s CLOUD
            ("Ollama Llama2 (local)", self._query_llama2),          # 16.10s OFFLINE
        ]

    # ============================================================
    # MULTI-MODEL ORCHESTRATION (Extractor → Analyzer → Enricher → Explainer)
    # ============================================================

    def _build_handoff_payload(self, role: str, task: str, question: str, previous: Dict, output_format: str = "structured_json", confidence_threshold: float = 0.7) -> Dict:
        return {
            "original_question": question,
            "my_role": role,
            "my_task": task,
            "context_from_previous": previous,
            "output_format": output_format,
            "confidence_threshold": confidence_threshold
        }

    def _extract_manifest(self, question: str, file_context: str) -> Tuple[str, Dict]:
        """Run extractor (fast model) to produce manifest of document structure."""
        extractor_prompt = self._build_handoff_payload(
            role="extractor",
            task="Estrai struttura documento: pagine, sezioni, immagini, tabelle, testo chiave. Restituisci manifest JSON compatto.",
            question=question,
            previous={"raw_text_excerpt": file_context[:4000] if file_context else ""},
            output_format="structured_json"
        )

        prompt = json.dumps(extractor_prompt, ensure_ascii=True)

        # Prefer Groq fast; fallback HF
        self._emit_stream("routing", "Seleziono modello veloce per estrazione (Groq → HF fallback)")
        text, success, metadata = self._query_groq(prompt)
        model_used = metadata.get("model") if success else "Groq (fallback HF)"
        if not success or not text:
            self._emit_stream("routing", "Groq non disponibile, passo a HuggingFace")
            text, success, metadata = self._query_huggingface(prompt)
            model_used = metadata.get("model") if success else "HuggingFace"

        manifest = {}
        if success and text:
            try:
                manifest = json.loads(text)
            except Exception:
                manifest = {"raw_manifest": text[:2000]}

        # Basic enrichment if manifest empty
        if not manifest:
            manifest = {
                "pages": file_context.count("PAGE"),
                "images_detected": 0,
                "tables_detected": 0,
                "summary": text[:800] if text else file_context[:800]
            }

        return text or "", {"manifest": manifest, "model": model_used, "metadata": metadata}

    def _analyze_content(self, question: str, manifest: Dict, file_context: str, preferred: str = "Mistral") -> Tuple[str, Dict]:
        analyzer_prompt = self._build_handoff_payload(
            role="analyzer",
            task="Analisi profonda del documento, connessioni, insight. Includi confidence score 0-1.",
            question=question,
            previous={"manifest": manifest, "raw_text_excerpt": file_context[:6000]},
            output_format="structured_json",
            confidence_threshold=0.75
        )
        prompt = json.dumps(analyzer_prompt, ensure_ascii=True)

        # Preferred analyzer: Mistral; fallback Gemini if available
        self._emit_stream("analyzing", "Analisi profonda con modello avanzato (Mistral → Gemini)")
        text, success, metadata = self._query_mistral(prompt)
        model_used = metadata.get("model") if success else "Mistral"
        if (not success or not text) and os.getenv("GOOGLE_API_KEY"):
            self._emit_stream("analyzing", "Fallback su Gemini per analisi")
            text, success, metadata = self._query_gemini(prompt)
            model_used = metadata.get("model") if success else "Gemini"

        if not text:
            text = "{}"

        try:
            analysis = json.loads(text)
        except Exception:
            analysis = {"analysis_text": text[:4000]}

        self.active_backend = model_used
        self.last_response_metadata = metadata or {}

        return text, {"analysis": analysis, "model": model_used, "metadata": metadata}

    def _enrich_context(self, analysis: Dict, question: str) -> Tuple[str, Dict]:
        enrichment_prompt = self._build_handoff_payload(
            role="enricher",
            task="Arricchisci con collegamenti esterni, contesto storico/artistico, riferimenti.",
            question=question,
            previous={"key_points": analysis},
            output_format="structured_json",
            confidence_threshold=0.7
        )
        prompt = json.dumps(enrichment_prompt, ensure_ascii=True)

        self._emit_stream("enriching", "Arricchimento con Gemini (web grounding)")
        text, success, metadata = self._query_gemini(prompt)
        model_used = metadata.get("model") if success else "Gemini"

        enrichment = {}
        if success and text:
            try:
                enrichment = json.loads(text)
            except Exception:
                enrichment = {"enrichment_text": text[:3000]}

        return text or "", {"enrichment": enrichment, "model": model_used, "metadata": metadata}

    def _explain_to_student(self, question: str, analysis: Dict, enrichment: Dict) -> Tuple[str, Dict]:
        explainer_prompt = self._build_handoff_payload(
            role="explainer",
            task="Sintesi chiara per studente ABA: sezioni, confronti, mappe concettuali. Usa tono accademico ma conciso.",
            question=question,
            previous={"analysis": analysis, "enrichment": enrichment},
            output_format="structured_json",
            confidence_threshold=0.7
        )
        prompt = json.dumps(explainer_prompt, ensure_ascii=True)

        self._emit_stream("composing", "Composizione risposta finale (Groq → Mistral)")
        text, success, metadata = self._query_groq(prompt)
        model_used = metadata.get("model") if success else "Groq"
        if not success or not text:
            text, success, metadata = self._query_mistral(prompt)
            model_used = metadata.get("model") if success else "Mistral"

        if not text:
            text = "Risposta non disponibile."

        self.active_backend = model_used
        self.last_response_metadata = metadata or {}

        return text, {"model": model_used, "metadata": metadata}

    def orchestrate_multimodel(self, question: str, file_context: str = "", enable_enrichment: bool = True, emit_callback: Optional[Callable[[str, str, Optional[str], Optional[Dict]], None]] = None) -> Dict:
        """Orchestrate extractor→analyzer→enricher(optional)→explainer.

        Returns dict with final_text, quality (ResponseQuality), steps metadata.
        """
        self.stream_callback = emit_callback

        # EXTRACTOR
        self._emit_stream("extracting", "Avvio estrazione struttura documento")
        extractor_text, extractor_info = self._extract_manifest(question, file_context)

        # ANALYZER
        self._emit_stream("handoff", "Passo il manifest all'analyzer", data={"manifest_keys": list(extractor_info.get("manifest", {}).keys())[:6]})
        analysis_text, analysis_info = self._analyze_content(question, extractor_info.get("manifest", {}), file_context)

        # QUALITY CHECK
        evaluator = ResponseQualityEvaluator()
        quality = evaluator.evaluate(analysis_text, question)
        self._emit_stream("quality_check", f"Parole: {quality.word_count} | Profondita: {quality.depth_score} | Confidenza: {quality.confidence}")

        # ENRICHER (conditional)
        enrichment_info = {"enrichment": {}, "model": None, "metadata": {}}
        if enable_enrichment and quality.confidence < 0.8:
            enrichment_text, enrichment_info = self._enrich_context(analysis_info.get("analysis", {}), question)
        else:
            self._emit_stream("enriching", "Arricchimento saltato (confidenza sufficiente)")

        # EXPLAINER
        explainer_text, explainer_info = self._explain_to_student(
            question,
            analysis_info.get("analysis", {}),
            enrichment_info.get("enrichment", {})
        )

        # FINAL QUALITY (on explainer output)
        final_quality = evaluator.evaluate(explainer_text, question)
        self._emit_stream("decision", final_quality.reason)

        return {
            "final_text": explainer_text,
            "quality": final_quality,
            "extractor": extractor_info,
            "analyzer": analysis_info,
            "enricher": enrichment_info,
            "explainer": explainer_info
        }
    
    def get_available_backends(self) -> Dict[str, str]:
        """Return list of available backends for UI selection"""
        return {
            i+1: name for i, (name, _) in enumerate(self._get_backend_chain())
        }
    
    def complete(self, prompt: str, **kwargs) -> CompletionResponse:
        """
        ✅ INTELLIGENT 8-TIER FALLBACK CHAIN WITH DYNAMIC SYSTEM PROMPTS
        Tries backends in optimal order, falls back automatically
        Supports multiple modalità (ragionamento, analisi, fashion_design, esami, presentazioni, documenti, generale)
        
        Parameters:
            forced_backend: Optional[str] - Force a specific backend (skip fallback chain)
                          Options: HuggingFace, Groq, Mistral, DeepSeek, Gemini, OpenRouter, Ollama-Local, Ollama-Cloud
        """
        # Extract modalità and forced_backend from kwargs
        modalita = kwargs.get("modalita", "generale")
        forced_backend = kwargs.get("forced_backend", None)
        
        # Build system prompt based on selected modalità
        system_prompt = self._get_system_prompt(modalita)
        
        # Construct final prompt with system instruction + user prompt
        final_prompt = system_prompt + "\n\n---\n\nUTENTE: " + prompt
        
        logger.info("\n" + "="*70)
        logger.info("ADVANCED REASONING & ANALYSIS LLM (v5 - ENTERPRISE)")
        logger.info("="*70)
        
        backends = self._get_backend_chain()
        
        # If forced_backend specified, use only that backend (no fallback)
        if forced_backend:
            logger.info(f"🔒 FORCED MODE: Using only '{forced_backend}'...\n")
            for backend_name, query_fn in backends:
                # More flexible matching - including partial matches
                fb = forced_backend.lower().replace("-", "").replace("_", "")
                bn = backend_name.lower().replace("-", "").replace("_", "").replace("(", "").replace(")", "").replace(".", "").replace("llama", "llama").replace("3.3", "33").replace("3", "3")
                backend_match = (
                    backend_name.lower() == forced_backend.lower() or
                    forced_backend.lower() in backend_name.lower() or
                    backend_name.lower().split()[0] == forced_backend.lower().split()[0] or  # "groq" matches "Groq llama-3.3-70b"
                    fb in bn
                )
                if backend_match:
                    logger.info(f"[FORCED] Using: {backend_name}")
                    text, success, metadata = query_fn(final_prompt)
                    if success and text:
                        self.active_backend = backend_name
                        self.active_model = metadata.get("model", "unknown")
                        self.last_response_metadata = metadata
                        logger.info(f"[OK] SUCCESS with forced backend: {backend_name}")
                        # Apply Italian language enforcement
                        text = self._force_italian_response(text)
                        return CompletionResponse(text=text, raw={"backend": backend_name, "metadata": metadata})
                    else:
                        # Clear previous metadata when forced backend fails
                        self.active_backend = "ERROR"
                        self.last_response_metadata = {"error": f"Backend '{forced_backend}' failed or returned empty response"}
                        error_msg = f"[ERROR] Forced backend '{forced_backend}' failed or returned empty response"
                        logger.error(error_msg)
                        return CompletionResponse(text=error_msg)
            
            # Backend not found
            available = [b[0] for b in backends]
            error_msg = f"[ERROR] Backend '{forced_backend}' not found. Available backends: {available}"
            logger.error(error_msg)
            return CompletionResponse(text=error_msg)
        
        # Normal fallback mode
        logger.info(f"Starting 8-tier intelligent fallback chain (modalita: {modalita})...\n")
        
        for idx, (backend_name, query_fn) in enumerate(backends, 1):
            logger.info(f"\n[{idx}/{len(backends)}] Trying: {backend_name}...")
            
            # Real-time callback to notify server of attempt
            if self.on_attempt:
                try:
                    self.on_attempt(idx, len(backends), backend_name)
                except Exception as e:
                    logger.debug(f"Callback error: {e}")
            
            # Log attempt
            self.attempts_log.append({
                "attempt": idx,
                "total": len(backends),
                "backend": backend_name
            })
            
            text, success, metadata = query_fn(final_prompt)
            
            if success and text:
                self.active_backend = backend_name
                self.active_model = metadata.get("model", "unknown")
                self.last_response_metadata = metadata
                
                logger.info(f"[OK] SUCCESS with {backend_name}")
                logger.info(f"   Model: {metadata.get('model')}")
                logger.info(f"   Provider: {metadata.get('provider', 'unknown')}")
                logger.info(f"   Latency: {metadata.get('latency')}")
                
                # Apply Italian language enforcement
                text = self._force_italian_response(text)
                
                return CompletionResponse(
                    text=text,
                    raw={"backend": backend_name, "metadata": metadata}
                )
            else:
                logger.warning(f"   [FALLBACK] Trying next backend...")
        
        # All backends failed
        error_msg = "[ERROR] ALL 8 BACKENDS FAILED - Check API keys and internet connection"
        logger.error(error_msg)
        return CompletionResponse(text=error_msg)
    
    def complete_with_vision(self, prompt: str, image_path: str, **kwargs) -> CompletionResponse:
        """
        VISION-CAPABLE COMPLETION - Analyzes both images AND text files
        - For images: Uses Gemini Vision API
        - For text files: Reads content and includes in prompt
        Falls back gracefully if file cannot be processed
        """
        modalita = kwargs.get("modalita", "generale")
        
        # Build system prompt based on modalità
        system_prompt = self._get_system_prompt(modalita)
        final_prompt = system_prompt + "\n\n---\n\nUTENTE: " + prompt
        
        logger.info("\n" + "="*70)
        logger.info("VISION-CAPABLE ANALYSIS (Gemini 2.5 Flash + Image)")
        logger.info("="*70)
        logger.info(f"Image: {image_path}")
        
        # Check if file exists and what type it is
        if not os.path.exists(image_path):
            logger.warning(f"   [ERROR] File not found: {image_path}")
            return self.complete(prompt, **kwargs)
        
        file_extension = os.path.splitext(image_path)[1].lower()
        is_image = file_extension in {'.jpg', '.jpeg', '.png', '.gif', '.webp', '.bmp'}
        is_text = file_extension in {'.txt', '.py', '.js', '.java', '.cpp', '.c', '.h', '.cs', '.rb', '.go', '.rs', '.ts', '.php', '.html', '.css', '.xml', '.json', '.yaml', '.yml', '.md', '.csv', '.log'}
        is_pdf = file_extension == '.pdf'
        
        # STRATEGY 1: Try image vision for image files
        if is_image:
            text, success, metadata = self._query_gemini_vision(final_prompt, image_path)
            
            if success and text:
                self.active_backend = "Google Gemini 2.5 Flash (Vision)"
                self.active_model = metadata.get("model", "unknown")
                self.last_response_metadata = metadata
                
                logger.info(f"[OK] Gemini Vision SUCCESS")
                logger.info(f"   Model: {metadata.get('model')}")
                # Apply Italian language enforcement
                text = self._force_italian_response(text)
                
                logger.info(f"   Image Info: {metadata.get('image_info')}")
                logger.info(f"   Latency: {metadata.get('latency')}")
                
                return CompletionResponse(
                    text=text,
                    raw={"backend": "Gemini Vision", "metadata": metadata}
                )
        
        # STRATEGY 2: For text files, read content and include in prompt
        if is_text:
            try:
                with open(image_path, 'r', encoding='utf-8') as f:
                    file_content = f.read()
                
                # Include file content in prompt
                file_name = os.path.basename(image_path)
                file_size_kb = os.path.getsize(image_path) / 1024
                
                # Truncate if too large
                max_chars = 8000
                if len(file_content) > max_chars:
                    file_content = file_content[:max_chars] + f"\n\n[... file truncated, total {len(file_content)} characters]"
                
                enriched_prompt = f"[File caricato: {file_name} ({file_size_kb:.1f} KB)]\n\nContenuto:\n```\n{file_content}\n```\n\n---\n\nAnalisi richiesta: {prompt}"
                final_prompt_with_content = system_prompt + "\n\n---\n\nUTENTE: " + enriched_prompt
                
                logger.info(f"   [TEXT-FILE] Included file content ({len(file_content)} chars)")
                logger.info(f"   [TEXT-FILE] Final prompt length: {len(final_prompt_with_content)} chars")
                
                # Use normal text completion with file content
                return self.complete(enriched_prompt, **kwargs)
                
            except Exception as e:
                logger.error(f"   [ERROR] Failed to read text file: {e}")
                return self.complete(prompt, **kwargs)
        
        # STRATEGY 3: For PDF, extract text and include in prompt
        if is_pdf:
            logger.info(f"   [PDF] Extracting text from PDF...")
            try:
                from pypdf import PdfReader
                
                # Extract text from PDF
                pdf_reader = PdfReader(image_path)
                pdf_text = ""
                num_pages = len(pdf_reader.pages)
                
                logger.info(f"   [PDF] Document has {num_pages} pages")
                
                # Extract text from all pages (increased limit for comprehensive extraction)
                # Limit is 50000 chars to capture full 47-page PDFs with 13+ slides
                max_chars = 50000  # Increased from 15000 to 50000 for better coverage
                chars_extracted = 0
                
                for page_num, page in enumerate(pdf_reader.pages, 1):
                    page_text = page.extract_text()
                    if page_text:
                        if chars_extracted + len(page_text) <= max_chars:
                            pdf_text += f"\n--- PAGE {page_num} ---\n{page_text}"
                            chars_extracted += len(page_text)
                        else:
                            # Truncate if we exceed max
                            remaining = max_chars - chars_extracted
                            if remaining > 0:
                                pdf_text += f"\n--- PAGE {page_num} (TRUNCATED) ---\n{page_text[:remaining]}"
                            break
                
                if not pdf_text.strip():
                    logger.warning(f"   [PDF] Could not extract text - PDF might be scanned/image-based")
                    # Try Gemini Vision as fallback for scanned PDFs
                    text, success, metadata = self._query_gemini_vision(final_prompt, image_path)
                    if success and text:
                        self.active_backend = "Google Gemini 2.5 Flash (Vision - PDF)"
                        self.active_model = metadata.get("model", "unknown")
                        self.last_response_metadata = metadata
                        logger.info(f"[OK] Gemini Vision handled scanned PDF")
                        text = self._force_italian_response(text)
                        return CompletionResponse(
                            text=text,
                            raw={"backend": "Gemini Vision (Scanned PDF)", "metadata": metadata}
                        )
                else:
                    # Include PDF content in prompt
                    file_name = os.path.basename(image_path)
                    file_size_kb = os.path.getsize(image_path) / 1024
                    
                    enriched_prompt = f"[File PDF caricato: {file_name} ({file_size_kb:.1f} KB, {num_pages} pagine)]\n\nContenuto estratto:\n```\n{pdf_text}\n```\n\n---\n\nAnalisi richiesta: {prompt}"
                    final_prompt_with_content = system_prompt + "\n\n---\n\nUTENTE: " + enriched_prompt
                    
                    logger.info(f"   [PDF] Extracted {chars_extracted} chars from {num_pages} pages")
                    logger.info(f"   [PDF] Final prompt length: {len(final_prompt_with_content)} chars")
                    
                    # Use normal text completion with PDF content
                    return self.complete(enriched_prompt, **kwargs)
                    
            except Exception as e:
                logger.error(f"   [ERROR] Failed to extract PDF text: {e}")
                logger.info(f"   [FALLBACK] Attempting Gemini Vision for PDF...")
                # Fallback to Gemini Vision
                text, success, metadata = self._query_gemini_vision(final_prompt, image_path)
                if success and text:
                    self.active_backend = "Google Gemini 2.5 Flash (Vision - PDF)"
                    self.active_model = metadata.get("model", "unknown")
                    self.last_response_metadata = metadata
                    text = self._force_italian_response(text)
                    return CompletionResponse(
                        text=text,
                        raw={"backend": "Gemini Vision (PDF Fallback)", "metadata": metadata}
                    )
                else:
                    return self.complete(prompt, **kwargs)
        
        # FALLBACK: Unknown file type, use regular completion
        logger.warning(f"   [FALLBACK] Cannot process file type '{file_extension}', using text-only completion...")
        return self.complete(prompt, **kwargs)
    
    def chat(self, messages: List[ChatMessage], **kwargs) -> ChatResponse:
        """Chat method (compatible with LlamaIndex)"""
        # Convert last user message to completion
        prompt = next(
            (m.content for m in reversed(messages) if m.role == "user"),
            "Ciao, come stai?"
        )
        
        response = self.complete(prompt)
        return ChatResponse(
            message=ChatMessage(role="assistant", content=response.text)
        )
    
    # ============================================================
    # ABSTRACT METHODS STUBS (required by LLM base class)
    # ============================================================
    
    async def achat(self, messages, **kwargs):
        """Async chat - stub"""
        raise NotImplementedError("Use sync chat() instead.")
    
    async def acomplete(self, prompt, **kwargs):
        """Async complete - stub"""
        raise NotImplementedError("Use sync complete() instead.")
    
    async def astream_chat(self, messages, **kwargs):
        """Async stream chat - stub"""
        raise NotImplementedError("Streaming not implemented.")
    
    async def astream_complete(self, prompt, **kwargs):
        """Async stream complete - stub"""
        raise NotImplementedError("Streaming not implemented.")
    
    def stream_chat(self, messages, **kwargs):
        """Stream chat - stub"""
        raise NotImplementedError("Use sync chat() instead.")
    
    def stream_complete(self, prompt, **kwargs):
        """Stream complete - stub"""
        raise NotImplementedError("Use sync complete() instead.")
    
    @property
    def metadata(self) -> LLMMetadata:
        """Metadata about this LLM"""
        return LLMMetadata(
            context_window=self.context_window,
            num_output=self.num_output,
            model_name="AdvancedReasoningLLM-v4-ULTIMATE",
        )


def get_analysis_llm(temperature: float = 0.3, max_tokens: int = 512) -> AdvancedReasoningLLM:
    """Factory function for creating analysis LLM"""
    return AdvancedReasoningLLM(temperature=temperature, max_tokens=max_tokens)


def get_reasoning_llm(temperature: float = 0.5) -> AdvancedReasoningLLM:
    """Factory function for reasoning mode"""
    return AdvancedReasoningLLM(temperature=temperature, max_tokens=1024)


def get_creative_llm(temperature: float = 0.9) -> AdvancedReasoningLLM:
    """Factory function for creative mode"""
    return AdvancedReasoningLLM(temperature=temperature, max_tokens=1024)


if __name__ == "__main__":
    # Quick test
    print("\n" + "="*70)
    print("🧪 Testing AdvancedReasoningLLM v4 (ULTIMATE - 8-tier system)")
    print("="*70)
    llm = get_analysis_llm()
    response = llm.complete("Spiega l'algoritmo quicksort in 3 righe")
    print(f"\n📝 Response:\n{response.text}\n")
    print(f"🔧 Backend Used: {llm.active_backend}")
    print(f"📊 Metadata: {llm.last_response_metadata}")
    print("="*70)
