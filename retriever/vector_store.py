"""
Vector Store - FAISS-based vector database with persistence
"""

import logging
import pickle
from pathlib import Path
from typing import List, Dict, Any, Optional

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class EmbeddingClient:
    """
    Wrapper around SentenceTransformer to generate embeddings.
    Explicitly exposes embedding dimension for downstream systems.
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    ):
        self.model = SentenceTransformer(model_name)
        self.dim = self.model.get_sentence_embedding_dimension()

        logger.info(f"Loaded embedding model: {model_name}")
        logger.info(f"Embedding dimension: {self.dim}")

    def embed(self, text: str) -> np.ndarray:
        """
        Generate embedding for a single text.
        """
        return self.model.encode(text, convert_to_numpy=True)

    def embed_batch(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for a batch of texts.
        """
        return self.model.encode(texts, convert_to_numpy=True)


class VectorStore:
    """
    FAISS-based vector store with in-memory index and optional persistence.
    """

    def __init__(
        self,
        dim: int,
        index_path: Optional[Path] = None
    ):
        self.dim = dim
        self.index = faiss.IndexFlatL2(dim)
        self.documents: List[Dict[str, Any]] = []

        self.index_path = index_path

        if self.index_path and self.index_path.exists():
            self._load()

        logger.info(f"Initialized FAISS index with dim={dim}")

    def add(
        self,
        embeddings: np.ndarray,
        documents: List[Dict[str, Any]]
    ):
        """
        Add embeddings and corresponding documents to the index.
        """
        if len(embeddings) != len(documents):
            raise ValueError("Embeddings and documents length mismatch")

        self.index.add(embeddings)
        self.documents.extend(documents)

        logger.info(f"Added {len(documents)} documents to vector store")

        if self.index_path:
            self._save()

    def search(
        self,
        query: str,
        top_k: int,
        embedding_fn
    ) -> List[Dict[str, Any]]:
        """
        Search the vector store for relevant documents.
        """
        query_embedding = embedding_fn(query).reshape(1, -1)

        distances, indices = self.index.search(query_embedding, top_k)

        results = []
        for idx in indices[0]:
            if idx < len(self.documents):
                results.append(self.documents[idx])

        logger.info(f"Retrieved {len(results)} documents from vector store")
        return results

    def _save(self):
        """
        Persist FAISS index and documents to disk.
        """
        data = {
            "index": faiss.serialize_index(self.index),
            "documents": self.documents
        }

        with open(self.index_path, "wb") as f:
            pickle.dump(data, f)

        logger.info(f"Vector store saved to {self.index_path}")

    def _load(self):
        """
        Load FAISS index and documents from disk.
        """
        with open(self.index_path, "rb") as f:
            data = pickle.load(f)

        self.index = faiss.deserialize_index(data["index"])
        self.documents = data["documents"]

        logger.info(f"Vector store loaded from {self.index_path}")
