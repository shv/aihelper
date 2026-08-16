from typing import Protocol

from openai import AsyncOpenAI
from openai.types import CreateEmbeddingResponse


class EmbeddingProvider(Protocol):
    async def embed(self, texts: list[str]) -> list[list[float]]: ...


class OpenAIEmbeddingProvider:
    def __init__(self, client: AsyncOpenAI, model: str) -> None:
        self._client = client
        self._model = model

    async def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            raise ValueError("Texts must not be empty")

        response: CreateEmbeddingResponse = await self._client.embeddings.create(
            model=self._model,
            input=texts,
            encoding_format="float",
        )

        ordered_items = sorted(response.data, key=lambda item: item.index)

        return [item.embedding for item in ordered_items]
