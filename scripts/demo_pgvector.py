"""
export DATABASE_URL='postgresql://aihelper:aihelper@127.0.0.1:5433/aihelper'
"""

import asyncio
import os

from openai import AsyncOpenAI

from app.rag.embeddings import OpenAIEmbeddingProvider
from app.rag.indexing import RagIndexer
from app.rag.store import PgVectorStore

from .demo_documents import CHUNKS

MODEL = "text-embedding-3-small"


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
