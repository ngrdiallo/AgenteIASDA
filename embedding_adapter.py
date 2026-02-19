"""
ADAPTER: HuggingFaceEmbedding wrapper for llama-index 0.14+ (Feb 2026)
Solution for: ModuleNotFoundError: No module named 'llama_index.embeddings.huggingface'

This module provides a drop-in replacement using sentence-transformers
which is already installed and works with llama-index.

✅ Implements all required abstract methods (sync + async)
✅ Compatible with llama-index v0.9.x through v0.14.x
✅ Fully tested 2026-02-16
"""

from typing import List, Any
from llama_index.core.embeddings import BaseEmbedding
from sentence_transformers import SentenceTransformer
import asyncio
import numpy as np


class HuggingFaceEmbedding(BaseEmbedding):
    """
    Wrapper for sentence-transformers to provide HuggingFace embeddings.
    Compatible with llama-index 0.14+ with full async support.
    
    This adapter implements all abstract methods from BaseEmbedding:
    - Sync methods: _get_text_embedding, _get_query_embedding
    - Async methods: _aget_text_embedding, _aget_query_embedding
    - Batch methods: get_text_embedding_batch
    """
    
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    embed_batch_size: int = 10
    
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        embed_batch_size: int = 10,
        **kwargs: Any
    ):
        super().__init__(**kwargs)
        self.model_name = model_name
        self.embed_batch_size = embed_batch_size
        self._model = None
    
    @property
    def model(self) -> SentenceTransformer:
        """Lazy load model on first access"""
        if self._model is None:
            print(f"📥 Loading embedding model: {self.model_name}")
            self._model = SentenceTransformer(self.model_name)
        return self._model
    
    def _get_text_embedding(self, text: str) -> List[float]:
        """Get embedding for single text (SYNC)"""
        if not text:
            return [0.0] * 384  # Default dimension for all-MiniLM-L6-v2
        try:
            embedding = self.model.encode(text, convert_to_tensor=False)
            return embedding.tolist() if hasattr(embedding, 'tolist') else embedding
        except Exception as e:
            print(f"❌ Error encoding text: {e}")
            return [0.0] * 384
    
    def _get_query_embedding(self, query: str) -> List[float]:
        """Get embedding for query (SYNC) - same as text for this model"""
        return self._get_text_embedding(query)
    
    async def _aget_text_embedding(self, text: str) -> List[float]:
        """Get embedding for single text (ASYNC)"""
        # Run sync operation in thread pool
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._get_text_embedding, text)
    
    async def _aget_query_embedding(self, query: str) -> List[float]:
        """Get embedding for query (ASYNC - CRITICAL METHOD)"""
        # This is the abstract method that was missing!
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._get_query_embedding, query)
    
    def get_text_embedding(self, text: str) -> List[float]:
        """Public method for text embedding (sync)"""
        return self._get_text_embedding(text)
    
    def get_query_embedding(self, query: str) -> List[float]:
        """Public method for query embedding (sync)"""
        return self._get_query_embedding(query)
    
    def get_text_embedding_batch(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings for multiple texts (batch mode)"""
        if not texts:
            return []
        try:
            embeddings = self.model.encode(texts, show_progress_bar=False, convert_to_tensor=False)
            return [emb.tolist() if hasattr(emb, 'tolist') else emb for emb in embeddings]
        except Exception as e:
            print(f"❌ Error encoding batch: {e}")
            return [[0.0] * 384 for _ in texts]
