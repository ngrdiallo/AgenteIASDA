#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VISION ANALYSIS ENGINE for ARTISTIC IMAGE ANALYSIS
Multi-modal vision models for deep artistic interpretation
Supports: DeepSeek-VL2, Gemini Vision, HuggingFace Vision

TARGET: ABA Bari students analyzing artworks, fashion design, compositions
CAPABILITIES:
  - Style analysis (Baroque, Renaissance, Futurism, etc.)
  - Composition breakdown (rule of thirds, focal points, balance)
  - Color theory interpretation
  - Artistic techniques identification
  - Fashion design element analysis
  - Metadata extraction (artist, period inference)
"""

import os
import json
import time
import logging
from typing import Tuple, Dict, Optional
from pathlib import Path

logger = logging.getLogger(__name__)

class VisionAnalysisEngine:
    """
    Multi-backend vision analysis for artistic image interpretation
    Falls back automatically if primary vision backend fails
    """
    
    def __init__(self):
        """Initialize vision analysis with available backends"""
        self.backends = self._load_backends()
        self.last_metadata = {}
    
    def _load_backends(self) -> Dict:
        """Load vision-capable backends from environment"""
        return {
            "deepseek_vl2": {
                "endpoint": "https://api.deepseek.com/v1/chat/completions",
                "model": "deepseek-vl2",
                "api_key": os.getenv("DEEPSEEK_API_KEY"),
                "priority": 1,
                "description": "DeepSeek Vision-Language (BEST for art analysis)"
            },
            "gemini_vision": {
                "endpoint": "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent",
                "model": "gemini-2.5-flash",
                "api_key": os.getenv("GOOGLE_API_KEY"),
                "priority": 2,
                "description": "Google Gemini 2.5 Flash Vision (1M token context)"
            },
            "huggingface_vision": {
                "endpoint": "https://api-inference.huggingface.co/models/Qwen/Qwen2-VL-7B-Instruct",
                "model": "Qwen2-VL-7B",
                "api_key": os.getenv("HF_TOKEN"),
                "priority": 3,
                "description": "HuggingFace Vision-Language (Open-source)"
            }
        }
    
    def analyze_artwork(self, image_path: str, analysis_depth: str = "standard") -> Tuple[str, Dict]:
        """
        Analyze artwork with artistic focus
        
        Args:
            image_path: Path to image file
            analysis_depth: "quick" (basic), "standard" (detailed), "deep" (extensive)
        
        Returns:
            Tuple[analysis_text, metadata]
        """
        
        # Validate image exists
        if not Path(image_path).exists():
            return f"[ERROR] Image not found: {image_path}", {"error": "file_not_found"}
        
        # Get image metadata
        file_size = Path(image_path).stat().st_size / 1024  # KB
        
        # Build artistic analysis prompt
        prompts = {
            "quick": "Analizza rapidamente questo dipinto: stile, colori principali, periodo storico stimato.",
            "standard": """Analizza questo capolavoro artistico in profondità:
1. STILE ARTISTICO: Movimento, periodo, influenze
2. COMPOSIZIONE: Regola dei terzi, punti focali, equilibrio
3. COLORI: Palette, contrasti, significato psicologico
4. TECNICHE: Pennellate, texture, uso della luce
5. CONTESTO: Possibile artista, periodo, scuola

Rispondi in italiano.""",
            "deep": """Analisi artistica ESTENSIVA. Rispondi in italiano dettagliato:

## ANALISI FORMALE
- Composizione e struttura geometrica
- Uso dello spazio e prospettiva
- Simmetria e asimmetria
- Movimento visivo e focal points

## TEORIA DEL COLORE
- Palette dominante e accenti
- Contrasti cromatici (complementari, analoghi)
- Temperatura colore (calda/fredda)
- Significato simbolico dei colori

## TECNICHE E STILE
- Movimento artistico (Rinascimento, Barocco, Impressionismo, Moderno, etc.)
- Stile dell'artista (se riconoscibile)
- Tecniche pittoriche (olio, acquerello, etc.)
- Dettagli di esecuzione

## CONTESTO STORICO-CULTURALE
- Periodo storico stimato
- Regione geografica/scuola
- Influenze e movimento artistico
- Significato iconografico

## IMPATTO ESTETICO
- Effetto emotivo
- Uso della luce e ombra
- Bilanciamento visivo
- Originalità e innovazione tecnica

Analisi per: studenti di Design/Arti ABA Bari."""
        }
        
        prompt = prompts.get(analysis_depth, prompts["standard"])
        
        # Try backends in priority order
        metadata = {
            "image_path": image_path,
            "file_size_kb": file_size,
            "analysis_depth": analysis_depth,
            "attempt_num": 0,
            "total_backends": len(self.backends)
        }
        
        for backend_name, backend_config in sorted(
            self.backends.items(), 
            key=lambda x: x[1]["priority"]
        ):
            if not backend_config["api_key"]:
                logger.info(f"[SKIP] {backend_name}: No API key configured")
                continue
            
            metadata["attempt_num"] += 1
            logger.info(f"[{metadata['attempt_num']}/{len(self.backends)}] Trying {backend_name}...")
            
            try:
                if "deepseek" in backend_name:
                    result, success = self._analyze_deepseek_vl2(image_path, prompt, backend_config)
                elif "gemini" in backend_name:
                    result, success = self._analyze_gemini_vision(image_path, prompt, backend_config)
                elif "huggingface" in backend_name:
                    result, success = self._analyze_huggingface_vision(image_path, prompt, backend_config)
                else:
                    continue
                
                if success and result:
                    metadata["backend"] = backend_name
                    metadata["backend_description"] = backend_config["description"]
                    self.last_metadata = metadata
                    return result, metadata
                
            except Exception as e:
                logger.warning(f"[FAIL] {backend_name}: {str(e)}")
                continue
        
        return "[ERROR] All vision backends failed - check API keys", metadata
    
    def _analyze_deepseek_vl2(self, image_path: str, prompt: str, config: Dict) -> Tuple[str, bool]:
        """Analyze with DeepSeek-VL2 Vision-Language model"""
        try:
            import requests
            from pathlib import Path
            import base64
            
            # Read and encode image
            with open(image_path, "rb") as f:
                image_data = base64.b64encode(f.read()).decode("utf-8")
            
            file_ext = Path(image_path).suffix.lower()
            media_type_map = {
                ".jpg": "image/jpeg",
                ".jpeg": "image/jpeg",
                ".png": "image/png",
                ".gif": "image/gif",
                ".webp": "image/webp"
            }
            media_type = media_type_map.get(file_ext, "image/jpeg")
            
            # DeepSeek API call with vision
            headers = {
                "Authorization": f"Bearer {config['api_key']}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": "deepseek-vl2",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:{media_type};base64,{image_data}"
                                }
                            },
                            {
                                "type": "text",
                                "text": prompt
                            }
                        ]
                    }
                ],
                "max_tokens": 2000,
                "temperature": 0.7
            }
            
            start = time.time()
            response = requests.post(config["endpoint"], json=payload, headers=headers, timeout=60)
            latency = time.time() - start
            
            if response.status_code == 200:
                data = response.json()
                analysis = data["choices"][0]["message"]["content"]
                logger.info(f"[OK] DeepSeek-VL2 analysis complete ({latency:.2f}s)")
                self.last_metadata["latency"] = f"{latency:.2f}s"
                return analysis, True
            else:
                logger.error(f"DeepSeek API error {response.status_code}")
                return "", False
                
        except Exception as e:
            logger.error(f"DeepSeek-VL2 analysis failed: {e}")
            return "", False
    
    def _analyze_gemini_vision(self, image_path: str, prompt: str, config: Dict) -> Tuple[str, bool]:
        """Analyze with Google Gemini Vision"""
        try:
            import google.generativeai as genai
            from pathlib import Path
            
            genai.configure(api_key=config["api_key"])
            
            # Load image
            image = genai.upload_file(image_path)
            
            # Create model and analyze
            model = genai.GenerativeModel("gemini-2.5-flash")
            
            start = time.time()
            response = model.generate_content([prompt, image])
            latency = time.time() - start
            
            analysis = response.text
            logger.info(f"[OK] Gemini Vision analysis complete ({latency:.2f}s)")
            self.last_metadata["latency"] = f"{latency:.2f}s"
            return analysis, True
            
        except Exception as e:
            logger.error(f"Gemini Vision analysis failed: {e}")
            return "", False
    
    def _analyze_huggingface_vision(self, image_path: str, prompt: str, config: Dict) -> Tuple[str, bool]:
        """Analyze with HuggingFace Vision-Language model"""
        try:
            import requests
            
            # Read image as base64
            with open(image_path, "rb") as f:
                image_data = f.read()
            
            headers = {"Authorization": f"Bearer {config['api_key']}"}
            
            # HuggingFace Inference API
            payload = {
                "inputs": {
                    "text": prompt,
                    "image": image_data
                }
            }
            
            start = time.time()
            response = requests.post(
                config["endpoint"],
                headers=headers,
                json=payload,
                timeout=60
            )
            latency = time.time() - start
            
            if response.status_code == 200:
                result = response.json()
                analysis = result[0].get("generated_text", "")
                logger.info(f"[OK] HuggingFace Vision analysis complete ({latency:.2f}s)")
                self.last_metadata["latency"] = f"{latency:.2f}s"
                return analysis, True
            else:
                logger.error(f"HuggingFace API error {response.status_code}")
                return "", False
                
        except Exception as e:
            logger.error(f"HuggingFace Vision analysis failed: {e}")
            return "", False
    
    def generate_artistic_image(self, prompt: str, style: str = "realistic") -> Tuple[str, Dict]:
        """
        Generate artistic image using Flux 2 Max (COMPLETELY FREE)
        
        Args:
            prompt: Description of image to generate
            style: artistic style descriptor
        
        Returns:
            Tuple[image_url, metadata]
        """
        try:
            import requests
            import urllib.parse
            
            # Pollinations API - COMPLETELY FREE, NO AUTH NEEDED
            full_prompt = f"{prompt}, style: {style}"
            encoded_prompt = urllib.parse.quote(full_prompt)
            
            # Direct image generation
            image_url = f"https://image.pollinations.ai/prompt/{encoded_prompt}"
            
            # Verify endpoint is responsive
            start = time.time()
            response = requests.head(image_url, timeout=10)
            latency = time.time() - start
            
            metadata = {
                "generator": "Flux 2 Max (Pollinations)",
                "model": "flux-2-max",
                "free_tier": True,
                "api_key_required": False,
                "latency": f"{latency:.2f}s",
                "image_url": image_url,
                "prompt": prompt,
                "style": style
            }
            
            logger.info(f"[OK] Flux 2 Max image generation ready - {image_url}")
            return image_url, metadata
            
        except Exception as e:
            logger.error(f"Image generation failed: {e}")
            return "", {"error": str(e)}


def get_vision_engine() -> VisionAnalysisEngine:
    """Factory function to get Vision Analysis Engine instance"""
    return VisionAnalysisEngine()
