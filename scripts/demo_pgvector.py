"""
export DATABASE_URL='postgresql://aihelper:aihelper@127.0.0.1:5433/aihelper'
"""

import asyncio
import os

from openai import AsyncOpenAI

from app.rag.embeddings import OpenAIEmbeddingProvider
from app.rag.indexing import RagIndexer
from app.rag.models import DocumentChunk
from app.rag.store import PgVectorStore

MODEL = "text-embedding-3-small"

CHUNKS = [
    DocumentChunk(
        id="gkl-cabinet",
        title="Крепление тяжёлого шкафа к стене из ГКЛ",
        text=(
            "Тяжёлый навесной шкаф на стене из ГКЛ рекомендуется "
            "крепить к стойкам каркаса или заранее установленной закладной."
        ),
        metadata={"category": "gkl"},
    ),
    DocumentChunk(
        id="tile-waterproofing",
        title="Подготовка мокрой зоны перед укладкой плитки",
        text=(
            "Перед укладкой плитки в мокрой зоне основание очищают, "
            "грунтуют и выполняют гидроизоляцию."
        ),
        metadata={"category": "tile"},
    ),
    DocumentChunk(
        id="electricity-safety",
        title="Безопасность при работах с розетками",
        text=(
            "Перед работами с розеткой необходимо отключить автомат "
            "и проверить отсутствие напряжения измерительным прибором."
        ),
        metadata={"category": "electricity"},
    ),
    DocumentChunk(
        id="plumbing-leak",
        title="Первые действия при протечке",
        text=(
            "При протечке сначала перекрывают воду, затем определяют "
            "место повреждения соединения или трубы."
        ),
        metadata={"category": "plumbing"},
    ),
    DocumentChunk(
        id="laminate-gap",
        title="Компенсационный зазор при укладке ламината",
        text=(
            "При укладке ламината вдоль стен оставляют компенсационный "
            "зазор для температурного расширения покрытия."
        ),
        metadata={"category": "flooring"},
    ),
]


async def main() -> None:
    client = AsyncOpenAI()
    embedding_provider = OpenAIEmbeddingProvider(client, MODEL)
    store = PgVectorStore(os.environ["DATABASE_URL"])

    indexer = RagIndexer(
        embedding_provider,
        store,
        embedding_model=MODEL,
    )

    stats = await indexer.index(CHUNKS)
    print("Indexing:", stats)

    query = "Подготовка стен ванной перед облицовкой"
    [query_embedding] = await embedding_provider.embed([query])

    results = await store.search(
        query_embedding,
        top_k=3,
        embedding_model=MODEL,
    )

    print("Result 3:", results)

    results = await store.search(
        query_embedding,
        top_k=3,
        category="tile",
        embedding_model=MODEL,
    )

    print("Result 1:", results)


if __name__ == "__main__":
    asyncio.run(main())
