"""
FastAPI Application
Exposes ingestion and query endpoints for the RAG system.
"""
from llm.llm_client import SimpleLLMClient

import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List

from ingestion.chunking import SemanticChunker
from ingestion.document_loader import DocumentLoader
from retriever.vector_store import VectorStore, EmbeddingClient
from llm.rag_pipeline import RAGPipeline

# --------------------------------------------------
# App Setup
# --------------------------------------------------

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Enterprise RAG Platform",
    description="Production-grade Retrieval-Augmented Generation system",
    version="1.0.0"
)

# --------------------------------------------------
# Initialize Core Components (Singletons)
# --------------------------------------------------

document_loader = DocumentLoader()
chunker = SemanticChunker()

embedding_client = EmbeddingClient()
vector_store = VectorStore(dim=embedding_client.dim)

llm_client = SimpleLLMClient()


rag_pipeline = RAGPipeline(
    vector_store=vector_store,
    embedding_client=embedding_client,
    llm_client=llm_client
)

# --------------------------------------------------
# Request / Response Models
# --------------------------------------------------

class IngestRequest(BaseModel):
    file_path: str


class QueryRequest(BaseModel):
    question: str


class QueryResponse(BaseModel):
    question: str
    answer: str
    latency_seconds: float
    evaluation: dict


# --------------------------------------------------
# Endpoints
# --------------------------------------------------

@app.post("/ingest")
def ingest_documents(request: IngestRequest):
    """
    Load document, chunk it, and ingest into vector store.
    """
    try:
        document = document_loader.load(request.file_path)
        chunks = chunker.chunk(document.text, source=document.source)

        rag_pipeline.ingest(chunks)

        return {
            "status": "success",
            "num_chunks_ingested": len(chunks),
            "source": document.source
        }

    except Exception as e:
        logger.error(f"Ingestion failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query", response_model=QueryResponse)
def query_rag(request: QueryRequest):
    """
    Query the RAG system.
    """
    try:
        result = rag_pipeline.query(request.question)

        return {
            "question": result["question"],
            "answer": result["answer"],
            "latency_seconds": result["latency_seconds"],
            "evaluation": result["evaluation"]
        }

    except Exception as e:
        logger.error(f"Query failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
        @app.get("/")
def root():
    return {
        "service": "Enterprise RAG Platform",
        "status": "running",
        "docs": "/docs"
    }
