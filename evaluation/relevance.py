"""
Relevance Evaluation
Measures semantic similarity between answer and retrieved context.
"""

import logging
from typing import List
import numpy as np

logger = logging.getLogger(__name__)


class RelevanceEvaluator:
    """
    Computes relevance score using cosine similarity
    between answer and retrieved context embeddings.
    """

    def __init__(self, embedding_client):
        self.embedding_client = embedding_client

    def score(self, answer: str, contexts: List[str]) -> float:
        """
        Args:
            answer: Generated LLM answer
            contexts: Retrieved document texts

        Returns:
            Relevance score between 0 and 1
        """
        if not answer or not contexts:
            logger.warning("Empty answer or context for relevance scoring")
            return 0.0

        try:
            answer_emb = self.embedding_client.embed(answer)
            context_embs = self.embedding_client.embed_batch(contexts)

            similarities = self._cosine_similarity(answer_emb, context_embs)
            score = float(np.mean(similarities))

            logger.info(f"Relevance score: {score:.3f}")
            return round(score, 3)

        except Exception as e:
            logger.error(f"Relevance scoring failed: {e}")
            return 0.0

    @staticmethod
    def _cosine_similarity(vec, matrix):
        vec_norm = np.linalg.norm(vec)
        matrix_norm = np.linalg.norm(matrix, axis=1)
        dot = matrix @ vec
        return dot / (matrix_norm * vec_norm + 1e-8)
