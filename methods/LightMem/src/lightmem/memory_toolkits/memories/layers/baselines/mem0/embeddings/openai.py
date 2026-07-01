import logging
import os
import time
import warnings
from typing import Literal, Optional

from openai import OpenAI

from mem0.configs.embeddings.base import BaseEmbedderConfig
from mem0.embeddings.base import EmbeddingBase


token_logger = logging.getLogger("mem0.token")


def _token_logging_enabled() -> bool:
    enabled = os.getenv("MEM0_TOKEN_LOG", "").lower() in {"1", "true", "yes", "on"}
    if enabled:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
        token_logger.setLevel(logging.INFO)
    return enabled


class OpenAIEmbedding(EmbeddingBase):
    def __init__(self, config: Optional[BaseEmbedderConfig] = None):
        super().__init__(config)

        self.config.model = self.config.model or "text-embedding-3-small"
        self.config.embedding_dims = self.config.embedding_dims or 1536

        api_key = self.config.api_key or os.getenv("OPENAI_API_KEY")
        base_url = (
            self.config.openai_base_url
            or os.getenv("OPENAI_API_BASE")
            or os.getenv("OPENAI_BASE_URL")
            or "https://api.openai.com/v1"
        )
        if os.environ.get("OPENAI_API_BASE"):
            warnings.warn(
                "The environment variable 'OPENAI_API_BASE' is deprecated and will be removed in the 0.1.80. "
                "Please use 'OPENAI_BASE_URL' instead.",
                DeprecationWarning,
            )

        self.client = OpenAI(api_key=api_key, base_url=base_url)

    def embed(self, text, memory_action: Optional[Literal["add", "search", "update"]] = None):
        """
        Get the embedding for the given text using OpenAI.

        Args:
            text (str): The text to embed.
            memory_action (optional): The type of embedding to use. Must be one of "add", "search", or "update". Defaults to None.
        Returns:
            list: The embedding vector.
        """
        text = text.replace("\n", " ")
        started_at = time.perf_counter()
        response = self.client.embeddings.create(
            input=[text],
            model=self.config.model,
            dimensions=self.config.embedding_dims,
        )
        elapsed_ms = (time.perf_counter() - started_at) * 1000
        if _token_logging_enabled():
            usage = getattr(response, "usage", None)
            token_logger.info(
                "[mem0.embedding] model=%s action=%s dims=%s input_chars=%d prompt_tokens=%s "
                "total_tokens=%s elapsed_ms=%.1f",
                self.config.model,
                memory_action,
                self.config.embedding_dims,
                len(text),
                getattr(usage, "prompt_tokens", None),
                getattr(usage, "total_tokens", None),
                elapsed_ms,
            )
        return response.data[0].embedding
