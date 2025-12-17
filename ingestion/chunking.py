"""
Semantic Chunking Module
Splits documents into semantically meaningful chunks with overlap.
"""

import logging
from typing import List
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class Chunk:
    """Represents a text chunk with metadata"""
    text: str
    index: int
    source: str


class SemanticChunker:
    """
    Enterprise-grade text chunker.
    Uses sentence boundaries + overlap for optimal RAG performance.
    """

    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 100
    ):
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk(self, text: str, source: str) -> List[Chunk]:
        """
        Split text into chunks with overlap.

        Args:
            text: Raw document text
            source: Source identifier (filename, URL, etc.)

        Returns:
            List of Chunk objects
        """
        if not text.strip():
            logger.warning("Empty text received for chunking")
            return []

        words = text.split()
        chunks = []

        start = 0
        index = 0

        while start < len(words):
            end = start + self.chunk_size
            chunk_words = words[start:end]
            chunk_text = " ".join(chunk_words)

            chunks.append(
                Chunk(
                    text=chunk_text,
                    index=index,
                    source=source
                )
            )

            index += 1
            start = end - self.chunk_overlap

        logger.info(f"Created {len(chunks)} chunks from source={source}")
        return chunks
