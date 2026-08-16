import pytest

from app.rag.indexing import IndexingStats, RagIndexer
from app.rag.models import DocumentChunk, EmbeddedChunk, IndexedChunkState


class FakeEmbeddingProvider:
    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    async def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(texts)
        return [[float(index), 1.0] for index, _ in enumerate(texts)]


class FakeIndexStore:
    def __init__(
        self,
        state: dict[str, IndexedChunkState] | None = None,
    ) -> None:
        self.state = state or {}
        self.get_index_state_calls: list[list[str]] = []
        self.upsert_calls: list[tuple[list[EmbeddedChunk], str]] = []

    async def get_index_state(
        self,
        chunk_ids: list[str],
    ) -> dict[str, IndexedChunkState]:
        self.get_index_state_calls.append(chunk_ids)
        return {
            chunk_id: self.state[chunk_id]
            for chunk_id in chunk_ids
            if chunk_id in self.state
        }

    async def upsert(
        self,
        chunks: list[EmbeddedChunk],
        *,
        embedding_model: str,
    ) -> None:
        self.upsert_calls.append((chunks, embedding_model))


CHUNK = DocumentChunk(
    id="tile-waterproofing",
    title="Подготовка мокрой зоны",
    text="Перед плиткой выполнить гидроизоляцию.",
    metadata={"category": "tile"},
)

MODEL = "test-embedding-model"


def indexed_state(
    chunk: DocumentChunk,
    *,
    content_hash: str | None = None,
    embedding_model: str = MODEL,
    metadata: dict[str, str] | None = None,
) -> IndexedChunkState:
    return IndexedChunkState(
        content_hash=content_hash or chunk.content_hash,
        embedding_model=embedding_model,
        metadata=chunk.metadata if metadata is None else metadata,
    )


@pytest.mark.asyncio
async def test_indexes_new_chunk() -> None:
    embedding_provider = FakeEmbeddingProvider()
    store = FakeIndexStore()
    indexer = RagIndexer(
        embedding_provider,
        store,
        embedding_model=MODEL,
    )

    stats = await indexer.index([CHUNK])

    assert stats == IndexingStats(total=1, indexed=1, skipped=0)
    assert store.get_index_state_calls == [[CHUNK.id]]
    assert embedding_provider.calls == [[CHUNK.embedding_text]]
    assert store.upsert_calls == [
        (
            [EmbeddedChunk(chunk=CHUNK, embedding=[0.0, 1.0])],
            MODEL,
        )
    ]


@pytest.mark.asyncio
async def test_skips_unchanged_chunk() -> None:
    embedding_provider = FakeEmbeddingProvider()
    store = FakeIndexStore(
        state={CHUNK.id: indexed_state(CHUNK)},
    )
    indexer = RagIndexer(
        embedding_provider,
        store,
        embedding_model=MODEL,
    )

    stats = await indexer.index([CHUNK])

    assert stats == IndexingStats(total=1, indexed=0, skipped=1)
    assert embedding_provider.calls == []
    assert store.upsert_calls == []


@pytest.mark.parametrize(
    "state",
    [
        IndexedChunkState(
            content_hash="outdated-content-hash",
            embedding_model=MODEL,
            metadata=CHUNK.metadata,
        ),
        IndexedChunkState(
            content_hash=CHUNK.content_hash,
            embedding_model="outdated-embedding-model",
            metadata=CHUNK.metadata,
        ),
        IndexedChunkState(
            content_hash=CHUNK.content_hash,
            embedding_model=MODEL,
            metadata={"category": "outdated"},
        ),
    ],
    ids=["content-changed", "model-changed", "metadata-changed"],
)
@pytest.mark.asyncio
async def test_reindexes_stale_chunk(state: IndexedChunkState) -> None:
    embedding_provider = FakeEmbeddingProvider()
    store = FakeIndexStore(state={CHUNK.id: state})
    indexer = RagIndexer(
        embedding_provider,
        store,
        embedding_model=MODEL,
    )

    stats = await indexer.index([CHUNK])

    assert stats == IndexingStats(total=1, indexed=1, skipped=0)
    assert embedding_provider.calls == [[CHUNK.embedding_text]]
    assert len(store.upsert_calls) == 1
    embedded_chunks, embedding_model = store.upsert_calls[0]
    assert embedded_chunks == [EmbeddedChunk(chunk=CHUNK, embedding=[0.0, 1.0])]
    assert embedding_model == MODEL


@pytest.mark.asyncio
async def test_indexes_only_changed_chunks_in_mixed_batch() -> None:
    new_chunk = DocumentChunk(
        id="drywall-partition",
        title="Перегородка из ГКЛ",
        text="Профиль устанавливается с заданным шагом.",
        metadata={"category": "drywall"},
    )
    embedding_provider = FakeEmbeddingProvider()
    store = FakeIndexStore(
        state={CHUNK.id: indexed_state(CHUNK)},
    )
    indexer = RagIndexer(
        embedding_provider,
        store,
        embedding_model=MODEL,
    )

    stats = await indexer.index([CHUNK, new_chunk])

    assert stats == IndexingStats(total=2, indexed=1, skipped=1)
    assert embedding_provider.calls == [[new_chunk.embedding_text]]
    assert store.upsert_calls == [
        (
            [EmbeddedChunk(chunk=new_chunk, embedding=[0.0, 1.0])],
            MODEL,
        )
    ]


@pytest.mark.asyncio
async def test_empty_batch_has_no_side_effects() -> None:
    embedding_provider = FakeEmbeddingProvider()
    store = FakeIndexStore()
    indexer = RagIndexer(
        embedding_provider,
        store,
        embedding_model=MODEL,
    )

    stats = await indexer.index([])

    assert stats == IndexingStats(total=0, indexed=0, skipped=0)
    assert store.get_index_state_calls == [[]]
    assert embedding_provider.calls == []
    assert store.upsert_calls == []


class InvalidEmbeddingProvider:
    async def embed(self, texts: list[str]) -> list[list[float]]:
        return []


@pytest.mark.asyncio
async def test_rejects_wrong_number_of_embeddings() -> None:
    store = FakeIndexStore()
    indexer = RagIndexer(
        InvalidEmbeddingProvider(),
        store,
        embedding_model=MODEL,
    )

    with pytest.raises(ValueError):
        await indexer.index([CHUNK])

    assert store.upsert_calls == []
