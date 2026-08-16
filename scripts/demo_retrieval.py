import asyncio
import json

from openai import AsyncOpenAI

from app.rag.embeddings import OpenAIEmbeddingProvider
from app.rag.models import DocumentChunk, EmbeddedChunk
from app.rag.search import search_top_k

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
    provider = OpenAIEmbeddingProvider(client)

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

    query = "Как безопасно повесить тяжёлый шкаф на гипсокартон?"
    query = "Что сделать перед заменой розетки?"
    query = "Подготовка стен ванной перед облицовкой"
    [query_embedding] = await provider.embed([query])

    results = search_top_k(
        query_embedding,
        index,
        top_k=3,
    )

    payload = {
        "query": query,
        "results": [
            {
                "id": result.chunk.id,
                "category": result.chunk.metadata["category"],
                "score": round(result.score, 4),
                "text": result.chunk.text,
            }
            for result in results
        ],
    }

    print(json.dumps(payload, ensure_ascii=False))


if __name__ == "__main__":
    asyncio.run(main())
