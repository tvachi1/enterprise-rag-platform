"""
Vector Store - FAISS-based vector database with persistence
"""
import logging
import pickle
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np

logger = logging.getLogger(__name__)


class VectorStore:
    """
    Production vector store using FAISS.
    Features:
    - Efficient similarity search
    - Persistence to disk
    - Metadata tracking
    """
    
    def __init__(
        self,
        embedding_dim: int = 768,
        index_type: str = "flat"
    ):
        """
        Args:
            embedding_dim: Dimension of embeddings (768 for BERT-base)
            index_type: FAISS index type ('flat' or 'ivf')
        """
        self.embedding_dim = embedding_dim
        self.index_type = index_type
        self.index = None
        self.documents = []  # Store metadata
        self._initialize_index()
        
    def _initialize_index(self):
        """Initialize FAISS index"""
        try:
            import faiss
            
            if self.index_type == "flat":
                # L2 distance (exact search)
                self.index = faiss.IndexFlatL2(self.embedding_dim)
            elif self.index_type == "ivf":
                # IVF for faster approximate search (production scale)
                quantizer = faiss.IndexFlatL2(self.embedding_dim)
                self.index = faiss.IndexIVFFlat(
                    quantizer,
                    self.embedding_dim,
                    100  # number of clusters
                )
            else:
                raise ValueError(f"Unknown index type: {self.index_type}")
                
            logger.info(f"Initialized FAISS index: {self.index_type}")
            
        except ImportError:
            logger.error("FAISS not installed. Run: pip install faiss-cpu")
            raise
    
    def add(
        self,
        embeddings: np.ndarray,
        documents: List[Dict[str, Any]]
    ):
        """
        Add embeddings and metadata to vector store
        
        Args:
            embeddings: (N, embedding_dim) array
            documents: List of document metadata dicts
        """
        if len(embeddings) != len(documents):
            raise ValueError("Embeddings and documents must have same length")
        
        # Train IVF index if needed
        if self.index_type == "ivf" and not self.index.is_trained:
            logger.info("Training IVF index...")
            self.index.train(embeddings)
        
        # Add to index
        self.index.add(embeddings.astype('float32'))
        self.documents.extend(documents)
        
        logger.info(f"Added {len(embeddings)} vectors. Total: {self.index.ntotal}")
    
    def search(
        self,
        query: str,
        top_k: int = 5,
        embedding_fn: Optional[callable] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for similar documents
        
        Args:
            query: Search query (string)
            top_k: Number of results
            embedding_fn: Function to embed query
            
        Returns:
            List of documents with scores
        """
        if self.index.ntotal == 0:
            logger.warning("Vector store is empty")
            return []
        
        # Embed query
        if embedding_fn is None:
            logger.warning("No embedding function provided, using placeholder")
            query_embedding = np.random.randn(1, self.embedding_dim)
        else:
            query_embedding = embedding_fn(query)
            if len(query_embedding.shape) == 1:
                query_embedding = query_embedding.reshape(1, -1)
        
        # Search
        distances, indices = self.index.search(
            query_embedding.astype('float32'),
            top_k
        )
        
        # Format results
        results = []
        for idx, distance in zip(indices[0], distances[0]):
            if idx < len(self.documents):
                doc = self.documents[idx].copy()
                # Convert L2 distance to similarity score (0-1)
                doc['score'] = float(1 / (1 + distance))
                results.append(doc)
        
        logger.info(f"Retrieved {len(results)} documents")
        return results
    
    def save(self, path: str):
        """Save index and metadata to disk"""
        try:
            import faiss
            
            path_obj = Path(path)
            path_obj.mkdir(parents=True, exist_ok=True)
            
            # Save FAISS index
            index_path = path_obj / "index.faiss"
            faiss.write_index(self.index, str(index_path))
            
            # Save metadata
            metadata_path = path_obj / "metadata.pkl"
            with open(metadata_path, 'wb') as f:
                pickle.dump({
                    'documents': self.documents,
                    'embedding_dim': self.embedding_dim,
                    'index_type': self.index_type
                }, f)
            
            logger.info(f"Saved vector store to {path}")
            
        except Exception as e:
            logger.error(f"Failed to save vector store: {e}")
            raise
    
    def load(self, path: str):
        """Load index and metadata from disk"""
        try:
            import faiss
            
            path_obj = Path(path)
            
            # Load FAISS index
            index_path = path_obj / "index.faiss"
            self.index = faiss.read_index(str(index_path))
            
            # Load metadata
            metadata_path = path_obj / "metadata.pkl"
            with open(metadata_path, 'rb') as f:
                metadata = pickle.load(f)
                self.documents = metadata['documents']
                self.embedding_dim = metadata['embedding_dim']
                self.index_type = metadata['index_type']
            
            logger.info(f"Loaded vector store from {path}")
            logger.info(f"Total vectors: {self.index.ntotal}")
            
        except Exception as e:
            logger.error(f"Failed to load vector store: {e}")
            raise
    
    def clear(self):
        """Clear all data from vector store"""
        self._initialize_index()
        self.documents = []
        logger.info("Cleared vector store")


class EmbeddingClient:
    """
    Client for generating embeddings
    Production: Use sentence-transformers or OpenAI
    """
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model_name = model_name
        self._load_model()
    
    def _load_model(self):
        """Load embedding model"""
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(self.model_name)
            logger.info(f"Loaded embedding model: {self.model_name}")
        except ImportError:
            logger.error("sentence-transformers not installed")
            self.model = None
    
    def embed(self, text: str) -> np.ndarray:
        """Generate embedding for text"""
        if self.model is None:
            # Fallback: random embedding (for testing only)
            return np.random.randn(768)
        
        return self.model.encode(text)
    
    def embed_batch(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for multiple texts"""
        if self.model is None:
            return np.random.randn(len(texts), 768)
        
        return self.model.encode(texts)
