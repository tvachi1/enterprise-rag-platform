"""
LLM Client Interface
Simple deterministic LLM stub for local development and testing.
"""

class SimpleLLMClient:
    """
    Minimal LLM client used for development and testing.
    """

    def generate(self, prompt: str) -> str:
        # Very simple, deterministic response for demo purposes
        return (
            "Based on the provided context, here is a concise answer. "
            "This response is generated using a stub LLM client."
        )
