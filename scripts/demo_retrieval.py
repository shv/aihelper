import asyncio
import json

from openai import AsyncOpenAI

from app.rag.bm25 import Bm25Index
from app.rag.embeddings import OpenAIEmbeddingProvider
from app.rag.fusion import reciprocal_rank_fusion
from app.rag.models import DocumentChunk, EmbeddedChunk
from app.rag.search import search_top_k

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
    DocumentChunk(
        id="tile-adhesive-c2te-s1",
        title="Плиточный клей класса C2TE S1",
        text="Клей имеет классификацию C2TE S1.",
        metadata={"category": "tile"},
    ),
    DocumentChunk(
        id="tile-adhesive-c1",
        title="Плиточный клей класса C1",
        text="Клей имеет классификацию C1.",
        metadata={"category": "tile"},
    ),
]


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
