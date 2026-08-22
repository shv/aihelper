import psycopg
from pgvector import Vector
from pgvector.psycopg import register_vector_async
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from .models import DocumentChunk, EmbeddedChunk, IndexedChunkState, SearchResult

UPSERT_SQL = """
INSERT INTO document_chunks (
    id,
    title,
    content,
    metadata,
    embedding,
    content_hash,
    embedding_model
)
VALUES (
    %(id)s,
    %(title)s,
    %(content)s,
    %(metadata)s,
    %(embedding)s,
    %(content_hash)s,
    %(embedding_model)s
)
ON CONFLICT (id) DO UPDATE
SET
    title = EXCLUDED.title,
    content = EXCLUDED.content,
    metadata = EXCLUDED.metadata,
    embedding = EXCLUDED.embedding,
    content_hash = EXCLUDED.content_hash,
    embedding_model = EXCLUDED.embedding_model,
    updated_at = now()
WHERE
    document_chunks.content_hash IS DISTINCT FROM EXCLUDED.content_hash
    OR document_chunks.metadata IS DISTINCT FROM EXCLUDED.metadata
    OR document_chunks.embedding_model IS DISTINCT FROM EXCLUDED.embedding_model

"""

SEARCH_SQL = """
SELECT
    id,
    title,
    content,
    metadata,
    1 - (embedding <=> %(embedding)s) AS score
FROM document_chunks
WHERE
    embedding_model = %(embedding_model)s
    AND (
        %(category)s::text IS NULL
        OR metadata ->> 'category' = %(category)s
    )
ORDER BY embedding <=> %(embedding)s
LIMIT %(top_k)s
"""

INDEX_SQL = """
SELECT
    id,
    content_hash,
    embedding_model,
    metadata
FROM document_chunks
WHERE id = ANY(%(chunk_ids)s)
"""


BM25_SEARCH_SQL = """
SELECT
    id,
    title,
    content,
    metadata,
    -(
        (title || E'\\n' || content)
        <@> to_bm25query(
            %(query)s,
            'document_chunks_bm25_idx'
        )
    ) AS score
FROM document_chunks
WHERE
    (
        %(category)s::text IS NULL
        OR metadata ->> 'category' = %(category)s
    )
ORDER BY
    (title || E'\\n' || content)
    <@> to_bm25query(
        %(query)s,
        'document_chunks_bm25_idx'
    )
LIMIT %(top_k)s
"""


class PgVectorStore:
    def __init__(self, dsn: str) -> None:
        self._dsn = dsn

    async def upsert(
        self, chunks: list[EmbeddedChunk], *, embedding_model: str
    ) -> None:
        parameters: list[dict[str, object]] = []
        for embedded_chunk in chunks:
            parameters.append(
                {
                    "id": embedded_chunk.chunk.id,
                    "title": embedded_chunk.chunk.title,
                    "content": embedded_chunk.chunk.text,
                    "metadata": Jsonb(embedded_chunk.chunk.metadata),
                    "embedding": Vector(embedded_chunk.embedding),
                    "content_hash": embedded_chunk.chunk.content_hash,
                    "embedding_model": embedding_model,
                }
            )

        if parameters:
            async with await psycopg.AsyncConnection.connect(self._dsn) as connection:
                await register_vector_async(connection)
                async with connection.cursor() as cursor:
                    await cursor.executemany(UPSERT_SQL, parameters)

    async def search(
        self,
        query_embedding: list[float],
        *,
        embedding_model: str,
        top_k: int,
        category: str | None = None,
    ) -> list[SearchResult]:
        if top_k <= 0:
            raise ValueError("top_k must be positive")

        parameters = {
            "embedding": Vector(query_embedding),
            "category": category,
            "top_k": top_k,
            "embedding_model": embedding_model,
        }

        async with await psycopg.AsyncConnection.connect(self._dsn) as connection:
            await register_vector_async(connection)
            async with connection.cursor(row_factory=dict_row) as cursor:
                await cursor.execute(SEARCH_SQL, parameters)
                return [
                    SearchResult(
                        chunk=DocumentChunk(
                            id=row["id"],
                            title=row["title"],
                            text=row["content"],
                            metadata=row["metadata"],
                        ),
                        score=float(row["score"]),
                    )
                    for row in await cursor.fetchall()
                ]

    async def get_index_state(
        self, chunk_ids: list[str]
    ) -> dict[str, IndexedChunkState]:
        if not chunk_ids:
            return {}

        async with await psycopg.AsyncConnection.connect(self._dsn) as connection:
            await register_vector_async(connection)
            async with connection.cursor(row_factory=dict_row) as cursor:
                await cursor.execute(INDEX_SQL, {"chunk_ids": chunk_ids})
                return {
                    row["id"]: IndexedChunkState(
                        content_hash=row["content_hash"],
                        embedding_model=row["embedding_model"],
                        metadata=row["metadata"],
                    )
                    for row in await cursor.fetchall()
                }

    async def search_bm25(
        self, query: str, *, top_k: int, category: str | None = None
    ) -> list[SearchResult]:
        if not query.strip():
            raise ValueError("query must not be blank")

        if top_k <= 0:
            raise ValueError("top_k must be positive")

        parameters = {
            "query": query,
            "category": category,
            "top_k": top_k,
        }

        async with (
            await psycopg.AsyncConnection.connect(self._dsn) as connection,
            connection.cursor(row_factory=dict_row) as cursor,
        ):
            await cursor.execute(
                BM25_SEARCH_SQL,
                parameters,
            )

            return [
                SearchResult(
                    chunk=DocumentChunk(
                        id=row["id"],
                        title=row["title"],
                        text=row["content"],
                        metadata=row["metadata"],
                    ),
                    score=float(row["score"]),
                )
                for row in await cursor.fetchall()
            ]
