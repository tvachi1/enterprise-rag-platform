"""
Faithfulness Evaluation
Detects hallucinations by measuring grounding in retrieved context.
"""

import logging
from typing import List

logger = logging.getLogger(__name__)


class FaithfulnessEvaluator:
    """
    Measures how much of the answer is supported by retrieved context.
    """

    def score(self, answer: str, contexts: List[str]) -> float:
        """
        Args:
            answer: Generated LLM answer
            contexts: Retrieved document texts

        Returns:
            Faithfulness score between 0 and 1
        """
        if not answer or not contexts:
            logger.warning("Empty answer or context for faithfulness scoring")
            return 0.0

        context_text = " ".join(contexts).lower()
        answer_tokens = set(answer.lower().split())

        supported_tokens = [
            token for token in answer_tokens if token in context_text
        ]

        score = len(supported_tokens) / max(len(answer_tokens), 1)

        logger.info(f"Faithfulness score: {score:.3f}")
        return round(score, 3)
