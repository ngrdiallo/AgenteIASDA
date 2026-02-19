#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MODALITA_DETAILS - Complete mapping of 7 dynamic modalities for AgenteIA

Each modalità has specialized backend preferences and system prompts
Designed for ABA Bari students/faculty (Fashion Design, Arts, Design focus)
"""

MODALITA_CONFIG = {
    "generale": {
        "nome_ia": "Socrate.IA",
        "icona": "🤖",
        "titolo": "Assistenza Generale",
        "utilizzo": "Chat libera, promemoria, workflow, lista, organizzazione",
        "token_limit": 2048,
        "token_limit_display": "Illimitati (max 2048 per risposta)",
        "sottotitolo": "Per domande quotidiane, lista, organizzazione",
        "descrizione_completa": "Assistente universale per qualsiasi domanda. Perfetto per organizzazione, brainstorming, promemoria.",
        "backend_preferito": "HuggingFace (veloce) o Groq (bilanciato)",
        "specialita": ["Chat libera", "Brainstorming", "Organizzazione", "Promemoria", "Workflow"],
        "colore": "#1e40af"   # Blu
    },
    
    "ragionamento": {
        "nome_ia": "Aristotele.IA",
        "icona": "🧠",
        "titolo": "Ragionamento Logico",
        "utilizzo": "Problem-solving, step-by-step logico, argomentazione, critica",
        "token_limit": 2048,
        "token_limit_display": "Illimitati (max 2048 per risposta)",
        "sottotitolo": "Analisi logica strutturata di problemi complessi",
        "descrizione_completa": "Ragionamento step-by-step con premesse esplicite e conclusioni logiche. Perfetto per matematica, filosofia, analisi critica.",
        "backend_preferito": "DeepSeek (CoT) o Groq Llama (ragionamento)",
        "specialita": ["Analisi logica", "Step-by-step", "Argomentazione", "Risoluzione problemi", "Filosofia"],
        "colore": "#7c3aed"   # Viola
    },
    
    "analisi": {
        "nome_ia": "Curie.IA",
        "icona": "📊",
        "titolo": "Analisi Strutturale",
        "utilizzo": "Analizza documenti, testi, immagini, dati forniti. Estrai punti chiave, criticità",
        "token_limit": 4096,
        "token_limit_display": "4096 (documenti estesi)",
        "sottotitolo": "Analisi dettagliata di testi, documenti, dati strutturati",
        "descrizione_completa": "Analizza documenti complessi: PDF, articoli, rapporti. Estrae punti chiave, identifica lacune, crea schemi.",
        "backend_preferito": "Gemini (vision per documenti) o HuggingFace",
        "specialita": ["Analisi testi", "Estrazione dati", "Mappe mentali", "Schemi", "Ricerca critica"],
        "colore": "#0891b2"   # Ciano
    },
    
    "fashion_design": {
        "nome_ia": "Leonardo.IA",
        "icona": "👗",
        "titolo": "Fashion Design & Moda",
        "utilizzo": "Specializzato in moda, design, estetica. Analisi trend, consulenza design",
        "token_limit": 4096,
        "token_limit_display": "4096 (analisi estese)",
        "sottotitolo": "Esperto di Fashion Design, storia della moda, trend forecasting, modellistica",
        "descrizione_completa": "Specializzato in Fashion Design (ABA Bari program). Analizza collezioni, trend, storytelling moda, sostenibilità, tessuti.",
        "backend_preferito": "Gemini Vision (per analisi immagini moda) o Leonardo",
        "specialita": ["Trend forecasting", "Analisi collezioni", "Design sostenibile", "Modellistica", "Comunicazione brand", "Portfolio"],
        "colore": "#ec4899"   # Rosa
    },
    
    "esami": {
        "nome_ia": "Pascal.IA",
        "icona": "📚",
        "titolo": "Preparazione Esami",
        "utilizzo": "Schede di studio, domande simulate, mappe concettuali, argomentazione",
        "token_limit": 2048,
        "token_limit_display": "2048 (schede sintetiche)",
        "sottotitolo": "Genera schede, domande modello, mappe per ABA Bari (Storia Arte, Estetica, etc)",
        "descrizione_completa": "Prepara esami con schede sintetiche, domande simulate, mappe concettuali. Focus memoria e argomentazione.",
        "backend_preferito": "Groq Llama (veloce + preciso) o DeepSeek",
        "specialita": ["Schede studio", "Domande simulate", "Mappe mentali", "Argomentazione", "Riassunti", "Revisione"],
        "colore": "#06b6d4"   # Azzurro
    },
    
    "presentazioni": {
        "nome_ia": "Spielberg.IA",
        "icona": "🎬",
        "titolo": "Creazione Presentazioni",
        "utilizzo": "Struttura narrativa, layout visivo, storytelling, suggerimenti design slide",
        "token_limit": 3000,
        "token_limit_display": "3000 (slide + note relatore)",
        "sottotitolo": "Aiuta con struttura narrativa, design slide, gerarchia contenuto",
        "descrizione_completa": "Crea presentazioni vincenti: outline slide, suggerimenti design, bilanciamento contenuto teorico + visuale.",
        "backend_preferito": "Gemini (vision per layout) o Mistral",
        "specialita": ["Storytelling", "Outline slide", "Design visuale", "Gerarchia contenuto", "Note relatore", "Narrativa"],
        "colore": "#f59e0b"   # Ambra
    },
    
    "documenti": {
        "nome_ia": "Tesla.IA",
        "icona": "📄",
        "titolo": "Gestione Documenti",
        "utilizzo": "Analizza file caricati (PDF, PPTX, DOCX, immagini). Estrai, riassumi, converti",
        "token_limit": 6000,
        "token_limit_display": "6000 (documenti completi)",
        "sottotitolo": "Parsing, estrazione, OCR, riconoscimento layout, conversione formati",
        "descrizione_completa": "Analizza documenti caricati: PDF, PowerPoint, Word, immagini. Estrae testo, riconosce layout, identifica azioni.",
        "backend_preferito": "Gemini Vision (per OCR e layout) o HuggingFace",
        "specialita": ["OCR", "Estrazione testo", "Conversione", "Riconoscimento layout", "Metadata", "Batch processing"],
        "colore": "#8b5cf6"   # Viola chiaro
    }
}

def get_modalita_display(modalita_key: str) -> dict:
    """Get complete display config for a modalità"""
    return MODALITA_CONFIG.get(modalita_key, MODALITA_CONFIG["generale"])

def get_all_modalita_list() -> list:
    """Get list of all modalita in display order"""
    return [
        ("generale", "🤖 Assistenza Generale - Socrate.IA"),
        ("ragionamento", "🧠 Ragionamento Logico - Aristotele.IA"),
        ("analisi", "📊 Analisi Documenti - Curie.IA"),
        ("fashion_design", "👗 Fashion Design & Moda - Leonardo.IA"),
        ("esami", "📚 Preparazione Esami - Pascal.IA"),
        ("presentazioni", "🎬 Creazione Presentazioni - Spielberg.IA"),
        ("documenti", "📄 Gestione Documenti - Tesla.IA")
    ]

def format_modalita_card(modalita_key: str) -> str:
    """Format a modalità as a detailed card (for display in UI)"""
    config = get_modalita_display(modalita_key)
    
    return f"""
## {config['icona']} {config['nome_ia']} - {config['titolo']}

**Utilizzo:** {config['utilizzo']}
**Limiti Token:** {config['token_limit_display']}
**Sottotitolo:** {config['sottotitolo']}
**Backend consigliato:** {config['backend_preferito']}

**Specialità:**
{chr(10).join(f"- {s}" for s in config['specialita'])}
"""

if __name__ == "__main__":
    # Demo
    print("MODALITA_DETAILS Demo\n")
    for key, display_name in get_all_modalita_list():
        config = get_modalita_display(key)
        print(f"{config['icona']} {config['nome_ia']}: {config['sottotitolo']}")
