"""
Configurazione centralizzata per l'agente.
Gestisce API keys, modelli e parametri globali.
"""

import os
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

# Carica .env se esiste
load_dotenv()

# ============================================================
# CONFIGURAZIONE MODELLI LOCALI
# ============================================================
# ✅ NESSUNA API KEY NECESSARIA!
# ✅ Sistema 100% GRATUITO, LOCALE, OFFLINE
# ✅ Compatibile con i3 a basse risorse

# Embedding model (locale, HuggingFace, gratis)
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# LLM model (locale, TinyLlama, gratis, 1.1B parameters)
LLM_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

# Config modelli (per compatibilità)
MODELS_CONFIG = {
    "local": {
        "embedding": EMBEDDING_MODEL,
        "llm": LLM_MODEL,
        "free": True,
        "offline": True,
        "requires_api_keys": False
    }
}

DEFAULT_INDEX_MODEL = "local"
DEFAULT_CHAT_MODEL = "local"

# ============================================================
# CONFIGURAZIONE PROGETTO
# ============================================================

PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
STORAGE_DIR = PROJECT_ROOT / "storage"
LOGS_DIR = PROJECT_ROOT / "logs"
METADATA_FILE = PROJECT_ROOT / "pdf_metadata.json"

# Crea le cartelle se non esistono
DATA_DIR.mkdir(exist_ok=True)
STORAGE_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)

# ============================================================
# PARAMETRI RAG
# ============================================================

RAG_CONFIG = {
    "chunk_size": 1024,
    "chunk_overlap": 100,
    "top_k_retrieval": 5,  # Quanti documenti recuperare per domanda
    "temperature": 0.2,  # Più basso = più deterministico, buono per analisi
}

# ============================================================
# SISTEMA DI CITAZIONE
# ============================================================

CITATION_FORMAT = {
    "style": "inline",  # 'inline' [p.45] oppure 'footnote'
    "include_snippet": True,  # Includi estratto testuale breve
    "include_page": True,
    "include_chunk_id": True,
}

# ============================================================
# PROMPT DI SISTEMA
# ============================================================

SYSTEM_PROMPTS = {
    "art_historian": """Tu sei un professore universitario di Storia dell'Arte Moderna con expertise in:
- Analisi iconografica sistematica
- Genealogie stilistiche e influenze artistiche
- Metodologia della ricerca documentale

REGOLA FONDAMENTALE: Ogni affermazione DEVE essere supportata da citazioni verificabili dai documenti.

Formato citazioni: [📄 nomefile.pdf | p. XX | "citazione breve oppure descrizione visiva"]

Se un'informazione NON è nei documenti forniti, DEVI indicarlo esplicitamente.
NON inventare dati. Se è un'ipotesi, marcala come tale: [⚠️ Ipotesi basata su...]

Rispondi in italiano, con rigore accademico.""",

    "matrix_builder": """Tu sei un assistente specializzato nella costruzione di matrici di analisi sistematica.

Compito: Quando chiesto, estrai e struttura i dati dai documenti PDF in una tabella standardizzata.

La matrice deve contenere:
- File PDF analizzato
- Numero pagine
- Artisti menzionati (con date se presenti)
- Temi iconografici
- Opere descritte (titolo, data, collocazione)
- Influenze esplicitate

SEMPRE includi i riferimenti pagina per ogni dato."""
}

# ============================================================
# FUNZIONI HELPER
# ============================================================

# ============================================================
# FUNZIONI HELPER
# ============================================================

def validate_api_keys() -> bool:
    """
    ✅ NESSUNA VALIDAZIONE NECESSARIA!
    Sistema 100% locale non richiede API keys.
    """
    print("✅ Sistema locale - nessuna API key necessaria!")
    return True


def get_model_name(model_key: str) -> str:
    """Converte chiavi umane a nomi modello."""
    mapping = {
        "gemini_flash": MODELS_CONFIG["gemini"]["fast"],
        "gemini_pro": MODELS_CONFIG["gemini"]["pro"],
        "claude_haiku": MODELS_CONFIG["claude"]["haiku"],
        "claude_sonnet": MODELS_CONFIG["claude"]["sonnet"],
    }
    return mapping.get(model_key, model_key)


if __name__ == "__main__":
    print("📋 Configurazione caricata:")
    print(f"  Project root: {PROJECT_ROOT}")
    print(f"  Data dir: {DATA_DIR}")
    print(f"  Storage dir: {STORAGE_DIR}")
    print(f"  API Keys valid: {validate_api_keys()}")
