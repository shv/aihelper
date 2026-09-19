import asyncio
import json

from openai import AsyncOpenAI

from app.rag.bm25 import Bm25Index
from app.rag.demo_documents import CHUNKS
from app.rag.embeddings import OpenAIEmbeddingProvider
from app.rag.fusion import reciprocal_rank_fusion
from app.rag.models import EmbeddedChunk
from app.rag.search import search_top_k

MODEL = "text-embedding-3-small"


async def main() -> None:
    client = AsyncOpenAI()
    provider = OpenAIEmbeddingProvider(client, MODEL)

    chunk_embeddings = await provider.embed([chunk.embedding_text for chunk in CHUNKS])

    index = [
        EmbeddedChunk(
            chunk=chunk,
            embedding=embedding,
        )
        for chunk, embedding in zip(
            CHUNKS,
            chunk_embeddings,
            strict=True,
        )
    ]

    bm25_index = Bm25Index(CHUNKS, k1=1.2, b=0.75)

    queries = [
        "C2TE S1",
        "гипсокартонная мебельная навеска",
    ]

    comparsions = []

    for query in queries:
        # query = "Как безопасно повесить тяжёлый шкаф на гипсокартон?"
        # query = "Что сделать перед заменой розетки?"
        # query = "Подготовка стен ванной перед облицовкой"
        [query_embedding] = await provider.embed([query])

        vector_results = search_top_k(
            query_embedding,
            index,
            top_k=3,
        )

        bm25_results = bm25_index.search(query, top_k=3, category=None)

        hybrid_results = reciprocal_rank_fusion(
            [vector_results, bm25_results], rrf_k=60, top_k=3
        )

        payload = {
            "query": query,
            "vector": [
                {
                    "id": result.chunk.id,
                    "category": result.chunk.metadata["category"],
                    "score": round(result.score, 4),
                    "text": result.chunk.text,
                }
                for result in vector_results
            ],
            "bm25": [
                {
                    "id": result.chunk.id,
                    "category": result.chunk.metadata["category"],
                    "score": round(result.score, 4),
                    "text": result.chunk.text,
                }
                for result in bm25_results
            ],
            "hybrid": [
                {
                    "id": result.chunk.id,
                    "category": result.chunk.metadata["category"],
                    "score": round(result.score, 4),
                    "text": result.chunk.text,
                }
                for result in hybrid_results
            ],
        }
        comparsions.append(payload)

    print(json.dumps(comparsions, ensure_ascii=False))


if __name__ == "__main__":
    asyncio.run(main())
