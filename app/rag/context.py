import json

from app.rag.models import SearchResult


def build_rag_context(results: list[SearchResult]) -> str:
    payload = [
        {
            "id": result.chunk.id,
            "title": result.chunk.title,
            "text": result.chunk.text,
        }
        for result in results
    ]

    return json.dumps(payload, ensure_ascii=False, indent=2)
