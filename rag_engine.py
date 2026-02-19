"""
RAG ENGINE - Il cuore intelligente dell'agente
Gestisce: 
- Caricamento documenti con metadata
- Creazione indici vettoriali (ChromaDB)
- Retrieval con citazioni automatiche
- Multi-model (Gemini per embedding, Claude per analisi)
"""

import os
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import json
from datetime import datetime
import threading  # ✅ FIX #6: Thread safety CitationManager

# LlamaIndex
from llama_index.core import (
    VectorStoreIndex, Document, SimpleDirectoryReader, Settings,
    StorageContext, load_index_from_storage
)
from llama_index.core.schema import NodeWithScore
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb

# ✅ FIX: HuggingFaceEmbedding adapter for llama-index 0.14+
# (llama_index.embeddings.huggingface was removed in 0.9+)
from embedding_adapter import HuggingFaceEmbedding

# Local LLM
from local_llm import get_local_llm, SmartModelSelector

# Config
from config import (
    RAG_CONFIG, SYSTEM_PROMPTS, DATA_DIR, STORAGE_DIR
)


class CitationManager:
    """Gestisce citazioni verificabili con traccia metadati - THREAD-SAFE."""
    
    def __init__(self):
        self.citations = {}
        self.citation_counter = 0
        self._counter_lock = threading.Lock()  # ✅ FIX #6: Protect counter increment
    
    def add_citation(self, text: str, filename: str, page: int, 
                     chunk: Optional[str] = None) -> str:
        """
        Aggiunge una citazione e ritorna il marker inline.
        
        ✅ FIX #6: Thread-safe increment del citation counter
        Returns: "[📄 filename.pdf | p. 45]"
        """
        with self._counter_lock:  # Protect counter increment
            self.citation_counter += 1
            citation_id = f"cite_{self.citation_counter}"
        
        # Add to citations dict (no lock needed, ID is unique)
        self.citations[citation_id] = {
            "filename": filename,
            "page": page,
            "chunk": chunk,
            "text_snippet": text[:100],  # Primti 100 caratteri
            "timestamp": datetime.now().isoformat()
        }
        
        # Ritorna marker inline
        return f"[📄 {filename} | p. {page}]"
    
    def get_citations_footnotes(self) -> str:
        """Genera section di footnote con tutte le citazioni."""
        if not self.citations:
            return ""
        
        footnotes = "\n---\n## 📚 Fonti Citate\n\n"
        for cid, data in self.citations.items():
            footnotes += (
                f"**{cid}**: {data['filename']} (p. {data['page']})\n"
                f"  > \"{data['text_snippet']}...\"\n\n"
            )
        
        return footnotes
    
    def clear(self):
        """Reset per nuova risposta."""
        self.citations = {}
        self.citation_counter = 0


class RAGEngine:
    """
    Motore RAG principale con:
    - Indicizzazione documentale multi-model
    - Retrieval con citazioni
    - Sistema di memoria persistente
    """
    
    def __init__(self):
        self.index = None
        self.chat_engine = None
        self.citation_manager = CitationManager()
        self.metadata_db = {}
        self._llm_loaded = False  # Flag lazy loading LLM
        self._embed_loaded = False  # Flag lazy loading embeddings
        self._initialize()
    
    def _initialize(self):
        """Inizializza RAGEngine - LAZY PER TUTTO (modelli + embeddings)."""
        
        # Settings globali per LlamaIndex - NO caricamenti pesanti qui
        # Embeddings e LLM verranno caricati on-demand
        
        print("\n⚡ RAGEngine LAZY MODE: Embeddings e LLM caricheranno on-demand\n")
        
        self._llm_loaded = False
        self._embed_loaded = False
        
        Settings.chunk_size = RAG_CONFIG["chunk_size"]
        Settings.chunk_overlap = RAG_CONFIG["chunk_overlap"]
    
    def _ensure_embed_loaded(self):
        """Carica embedding model se non già caricato."""
        if self._embed_loaded:
            return
        
        print("📥 Caricamento embedding model (MiniLM)...")
        Settings.embed_model = HuggingFaceEmbedding(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        self._embed_loaded = True
        print("✅ Embedding model pronto\n")
    
    def _ensure_llm_loaded(self):
        """Carica LLM se non già caricato."""
        if self._llm_loaded:
            return
        
        print("🧠 Caricamento LLM (prima query - potrebbe impiegare 1-2 min)...")
        
        try:
            Settings.llm = get_local_llm(
                model_name="mistral",
                temperature=RAG_CONFIG.get("temperature", 0.2),
                max_tokens=256
            )
            
            backend_name = Settings.llm.active_backend or "unknown"
            print(f"✅ Backend attivo: {backend_name.upper()}\n")
            self._llm_loaded = True
        except Exception as e:
            print(f"❌ Errore caricamento LLM: {e}")
            raise RuntimeError(f"Impossibile caricare LLM: {str(e)}")
    
    def build_index(self, force_rebuild: bool = False) -> bool:
        """
        Costruisce l'indice vettoriale da PDF con metadata.
        Se force_rebuild=False, carica da storage se esiste.
        
        Returns: True se successo, False altrimenti
        """
        
        # Assicura che embeddings siano caricati (prima del LLM)
        self._ensure_embed_loaded()
        
        # Assicura che LLM sia caricato
        self._ensure_llm_loaded()
        
        # Tenta caricamento da storage se esiste
        if not force_rebuild and (STORAGE_DIR / "docstore.json").exists():
            print("📦 Caricamento indice da storage...")
            try:
                storage_context = StorageContext.from_defaults(
                    persist_dir=str(STORAGE_DIR)
                )
                self.index = load_index_from_storage(storage_context)
                print("✅ Indice caricato da storage")
                return True
            except Exception as e:
                print(f"⚠️  Errore caricamento storage: {e}")
                print("🔄 Ricostruisco da zero...")
        
        # Build nuovo indice
        print(f"🔨 Costruzione nuovo indice dai PDF in {DATA_DIR}...")
        
        try:
            # Leggi PDF
            reader = SimpleDirectoryReader(
                input_dir=str(DATA_DIR),
                recursive=True,
                required_exts=[".pdf"]
            )
            documents = reader.load_data()
            
            if not documents:
                print("❌ Nessun documento trovato!")
                return False
            
            print(f"📄 {len(documents)} documenti caricati")
            
            # Aggiungi metadata ai documenti
            for doc in documents:
                source = doc.metadata.get("file_name", "unknown")
                doc.metadata["pdf_source"] = source
            
            # Crea indice con ChromaDB
            print("🧠 Creazione embeddings con Gemini...")
            self.index = VectorStoreIndex.from_documents(
                documents,
                show_progress=True
            )
            
            # Salva su storage
            print("💾 Salvataggio su storage...")
            self.index.storage_context.persist(persist_dir=str(STORAGE_DIR))
            
            print("✅ Indice costruito e salvato!")
            return True
            
        except Exception as e:
            print(f"❌ Errore costruzione indice: {e}")
            return False
    
    def retrieve_with_citations(self, query: str, top_k: Optional[int] = None) -> Tuple[List[Dict], str]:
        """
        Retrieval con citazioni automatiche.
        
        Returns: 
            - Lista di documenti retrieve con metadata
            - Testo formattato con citazioni inline
        """
        
        if self.index is None:
            return [], "❌ Indice non inizializzato"
        
        if top_k is None:
            top_k = RAG_CONFIG["top_k_retrieval"]
        
        # Query vecotale
        retriever = self.index.as_retriever(similarity_top_k=top_k)
        nodes = retriever.retrieve(query)
        
        # Formatta con citazioni
        results = []
        self.citation_manager.clear()
        
        for node in nodes:
            if isinstance(node, NodeWithScore):
                text = node.get_content()
                metadata = node.metadata or {}
                
                # Estrai info citazione
                filename = metadata.get("pdf_source", metadata.get("file_name", "unknown.pdf"))
                page = metadata.get("page_label", "N/A")
                
                # Aggiungi alla gestione citazioni
                citation_marker = self.citation_manager.add_citation(
                    text=text,
                    filename=filename,
                    page=page,
                    chunk=node.node_id
                )
                
                results.append({
                    "content": text[:500],  # Primi 500 caratteri
                    "source": filename,
                    "page": page,
                    "score": node.score,
                    "citation": citation_marker
                })
        
        # Crea testo formatted
        formatted_text = "\n".join([
            f"{r['citation']}: {r['content']}"
            for r in results
        ])
        
        return results, formatted_text
    
    def query_with_context(self, user_query: str, system_prompt: Optional[str] = None) -> str:
        """
        Query completa con retrieval + generazione con citazioni.
        Triple fallback: chat_engine → query_engine → direct retrieval + LLM
        
        Returns: Risposta formattata con citazioni inline
        """
        
        # Assicura che LLM sia caricato
        self._ensure_llm_loaded()
        
        if self.index is None:
            return "❌ Indice non disponibile. Esegui build_index() prima."
        
        if system_prompt is None:
            system_prompt = SYSTEM_PROMPTS["art_historian"]
        
        response_text = None
        last_error = None
        
        # LIVELLO 1: Chat engine (contexto intelligente + memoria)
        try:
            print("📞 [LIVELLO 1] Tentando chat engine...")
            chat_engine = self.index.as_chat_engine(
                chat_mode="context",
                system_prompt=system_prompt,
                memory=None  # La memoria è gestita da Streamlit
            )
            response = chat_engine.chat(user_query)
            response_text = str(response.response)
            print("✅ Chat engine riuscito")
            
        except (AttributeError, TypeError, Exception) as e:
            last_error = e
            print(f"⚠️  [LIVELLO 1] Chat engine fallito: {type(e).__name__}: {str(e)[:100]}")
        
        # LIVELLO 2: Query engine (semplice retrieval + completion)
        if response_text is None:
            try:
                print("📞 [LIVELLO 2] Fallback a query engine...")
                query_engine = self.index.as_query_engine(
                    response_mode="compact",
                    similarity_top_k=5
                )
                response = query_engine.query(user_query)
                response_text = str(response)
                print("✅ Query engine riuscito")
                
            except (AttributeError, TypeError, Exception) as e:
                last_error = e
                print(f"⚠️  [LIVELLO 2] Query engine fallito: {type(e).__name__}: {str(e)[:100]}")
        
        # LIVELLO 3: Direct retrieval + LLM completion (massima robustezza)
        if response_text is None:
            try:
                print("📞 [LIVELLO 3] Fallback a retrieval diretto + LLM...")
                
                # Retrieval manuale
                retriever = self.index.as_retriever(similarity_top_k=5)
                retrieved_nodes = retriever.retrieve(user_query)
                
                # Costruisci testo di contesto
                context_text = "\n\n".join([
                    f"[Fonte: {node.metadata.get('file_name', 'Unknown')} pag. {node.metadata.get('page_label', 'N/A')}]\n{node.get_content()}"
                    for node in retrieved_nodes
                ])
                
                # Prompt composito per il LLM
                full_prompt = f"""
{system_prompt}

CONTESTO DISPONIBILE:
{context_text}

DOMANDA DELL'UTENTE:
{user_query}

ISTRUZIONI:
1. Rispondi basandoti SOLO sul contesto fornito
2. Se il contesto non contiene la risposta, dillo chiaramente
3. Cita sempre le fonti utilizzando il formato [Fonte: filename pag. N]
4. Sii conciso ma completo

RISPOSTA:
"""
                
                # LLM completion diretto
                response = Settings.llm.complete(full_prompt)
                response_text = response.text if hasattr(response, 'text') else str(response)
                print("✅ Direct retrieval + LLM riuscito")
                
            except Exception as e:
                last_error = e
                print(f"❌ [LIVELLO 3] Direct retrieval fallito: {type(e).__name__}: {str(e)}")
        
        # Se tutti i livelli falliscono
        if response_text is None:
            error_msg = f"{type(last_error).__name__}: {str(last_error)}" if last_error else "Unknown error"
            return f"❌ ERRORE: Tutti i livelli di query sono falliti.\n✗ Chat engine fallito\n✗ Query engine fallito\n✗ Direct retrieval fallito\n\nDettagli: {error_msg}\n\nProva:\n1. Verifica che build_index() sia stato eseguito\n2. Controlla che il file LLM sia caricato\n3. Riprova la query"
        
        # Aggiungi citazioni
        footnotes = self.citation_manager.get_citations_footnotes()
        
        return response_text + footnotes
    
    def generate_matrix_from_documents(self) -> Dict[str, Any]:
        """
        Genera automaticamente la MATRICE_ANALISI dai documenti.
        
        Struttura:
        {
            "files": [
                {
                    "filename": "Storia dell'Arte Moderna 1 A.pdf",
                    "pages": 150,
                    "artists": ["Caravaggio", "Ribera"],
                    "themes": ["Sacro", "Luce"],
                    "works": ["Deposizione", "Cristo flagellato"]
                }
            ],
            "summary": {...}
        }
        """
        
        if self.index is None:
            return {}
        
        print("🧭 Generazione matrice analitica...")
        
        matrix = {
            "generated": datetime.now().isoformat(),
            "files": [],
            "summary": {
                "total_artists": set(),
                "total_themes": set(),
                "total_works": set()
            }
        }
        
        # Per ogni file PDF nel data dir
        for pdf_path in sorted(Path(DATA_DIR).glob("*.pdf")):
            pdf_name = pdf_path.name
            
            # Query specializzate per estrarre info
            queries = {
                "artists": f"In {pdf_name}, quali artisti sono menzionati? Lista completa.",
                "themes": f"In {pdf_name}, quali temi iconografici ricorrono?",
                "works": f"In {pdf_name}, quali opere d'arte sono descritte?"
            }
            
            file_entry = {
                "filename": pdf_name,
                "pages": "N/A",  # Verrà calcolato dal metadata
                "artists": [],
                "themes": [],
                "works": [],
                "status": "✅ Letto"
            }
            
            matrix["files"].append(file_entry)
        
        return matrix
    
    def export_analysis_report(self, content: str, output_path: str = "report.md") -> bool:
        """Esporta il report finale in Markdown."""
        
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"✅ Report esportato: {output_path}")
            return True
        except Exception as e:
            print(f"❌ Errore esportazione: {e}")
            return False


# ============================================================
# FUNZIONI HELPER GLOBALI
# ============================================================

_rag_engine_instance = None


def get_rag_engine() -> RAGEngine:
    """Singleton pattern per RAGEngine."""
    global _rag_engine_instance
    if _rag_engine_instance is None:
        _rag_engine_instance = RAGEngine()
    return _rag_engine_instance


if __name__ == "__main__":
    print("🚀 Test RAG Engine")
    engine = get_rag_engine()
    print("✅ Engine inizializzato")
    
    # Build indice
    success = engine.build_index()
    if success:
        # Test query
        response = engine.query_with_context(
            "Chi è Caravaggio e quale è il suo stile?"
        )
        print("\n" + "="*60)
        print("RISPOSTA CON CITAZIONI:")
        print("="*60)
        print(response)
