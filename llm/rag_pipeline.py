"""
RAG Pipeline
End-to-end retrieval-augmented generation with metrics.
"""

import logging
import time
from typing import List, Dict, Any

from ingestion.chunking import Chunk
from retriever.vector_store import VectorStore, EmbeddingClient

logger = logging.getLogger(__name__)


class RAGPipeline:
    """
    Production-grade RAG pipeline.
    Handles retrieval, prompt construction, and inference.
    """

    def __init__(
        self,
        vector_store: VectorStore,
        embedding_client: EmbeddingClient,
        llm_client: Any,
        top_k: int = 5
    ):
        self.vector_store = vector_store
        self.embedding_client = embedding_client
        self.llm_client = llm_client
        self.top_k = top_k

    def ingest(self, chunks: List[Chunk]):
        """
        Ingest chunks into the vector store.
        """
        texts = [chunk.text for chunk in chunks]
        embeddings = self.embedding_client.embed_batch(texts)

        documents = [
            {
                "text": chunk.text,
                "source": chunk.source,
                "chunk_index": chunk.index
            }
            for chunk in chunks
        ]

        self.vector_store.add(embeddings, documents)
        logger.info(f"Ingested {len(chunks)} chunks into vector store")

    def query(self, question: str) -> Dict[str, Any]:
        """
        Execute RAG query and return answer with metrics.
        """
        start_time = time.time()

        retrieved_docs = self.vector_store.search(
            query=question,
            top_k=self.top_k,
            embedding_fn=self.embedding_client.embed
        )

        context = "\n\n".join(doc["text"] for doc in retrieved_docs)

        prompt = self._build_prompt(question, context)

        answer = self.llm_client.generate(prompt)

        latency = time.time() - start_time

        result = {
            "question": question,
            "answer": answer,
            "sources": retrieved_docs,
            "latency_seconds": round(latency, 3),
            "num_retrieved_docs": len(retrieved_docs)
        }

        logger.info(
            f"RAG query completed in {result['latency_seconds']}s "
            f"with {len(retrieved_docs)} documents"
        )

        return result

    def _build_prompt(self, question: str, context: str) -> str:
        """
        Build structured prompt for LLM.
        """
        return f"""
You are an enterprise AI assistant.
Answer the question using ONLY the provided context.
If the answer is not present, say you don't know.

Context:
{context}

Question:
{question}

Answer:
""".strip()
