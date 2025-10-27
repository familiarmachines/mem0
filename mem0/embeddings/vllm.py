import os
from typing import Literal, Optional
from openai import OpenAI

from mem0.configs.embeddings.base import BaseEmbedderConfig
from mem0.embeddings.base import EmbeddingBase


class VllmEmbedding(EmbeddingBase):
    def __init__(self, config: Optional[BaseEmbedderConfig] = None):
        super().__init__(config)

        self.config.model = self.config.model or "mixedbread-ai/mxbai-embed-xsmall-v1"
        self.config.embedding_dims = self.config.embedding_dims or 384
        self.config.vllm_base_url = self.config.vllm_base_url or os.getenv("VLLM_BASE_URL")
        self.client = OpenAI(api_key="dummy", base_url=self.config.vllm_base_url)

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
        return (
            self.client.embeddings.create(input=[text], model=self.config.model)
            .data[0]
            .embedding
        )
