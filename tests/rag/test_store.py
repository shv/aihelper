from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock

import psycopg
import pytest
from psycopg.rows import dict_row

from app.rag.models import DocumentChunk, SearchResult
from app.rag.store import BM25_SEARCH_SQL, PgVectorStore

DSN = "postgresql://test:test@127.0.0.1:5432/test"


def mock_database_connection(
    monkeypatch: pytest.MonkeyPatch,
    *,
    rows: list[dict[str, object]],
) -> tuple[AsyncMock, MagicMock, MagicMock]:
    cursor = MagicMock()
    cursor.__aenter__ = AsyncMock(return_value=cursor)
    cursor.__aexit__ = AsyncMock(return_value=None)
    cursor.execute = AsyncMock(return_value=None)
    cursor.fetchall = AsyncMock(return_value=rows)

    connection = MagicMock()
    connection.__aenter__ = AsyncMock(return_value=connection)
    connection.__aexit__ = AsyncMock(return_value=None)
    connection.cursor.return_value = cursor

    connect = AsyncMock(return_value=connection)
    monkeypatch.setattr(psycopg.AsyncConnection, "connect", connect)

    return connect, connection, cursor


@pytest.mark.asyncio
async def test_search_bm25_executes_parameterized_query_and_maps_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    connect, connection, cursor = mock_database_connection(
        monkeypatch,
        rows=[
            {
                "id": "tile-waterproofing",
                "title": "Подготовка мокрой зоны",
                "content": "Основание грунтуют и гидроизолируют.",
                "metadata": {"category": "tile"},
                "score": Decimal("3.3080638647"),
            }
        ],
    )
    store = PgVectorStore(DSN)

    results = await store.search_bm25(
        "плитка гидроизоляция",
        top_k=3,
        category="tile",
    )

    assert results == [
        SearchResult(
            chunk=DocumentChunk(
                id="tile-waterproofing",
                title="Подготовка мокрой зоны",
                text="Основание грунтуют и гидроизолируют.",
                metadata={"category": "tile"},
            ),
            score=float(Decimal("3.3080638647")),
        )
    ]
    connect.assert_awaited_once_with(DSN)
    connection.cursor.assert_called_once_with(row_factory=dict_row)
    cursor.execute.assert_awaited_once_with(
        BM25_SEARCH_SQL,
        {
            "query": "плитка гидроизоляция",
            "category": "tile",
            "top_k": 3,
        },
    )
    cursor.fetchall.assert_awaited_once_with()


@pytest.mark.asyncio
async def test_search_bm25_passes_none_category_explicitly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, _, cursor = mock_database_connection(monkeypatch, rows=[])
    store = PgVectorStore(DSN)

    results = await store.search_bm25(
        "C2TE S1",
        top_k=5,
        category=None,
    )

    assert results == []
    cursor.execute.assert_awaited_once_with(
        BM25_SEARCH_SQL,
        {
            "query": "C2TE S1",
            "category": None,
            "top_k": 5,
        },
    )


@pytest.mark.parametrize("query", ["", "   ", "\n\t"])
@pytest.mark.asyncio
async def test_search_bm25_rejects_blank_query(query: str) -> None:
    store = PgVectorStore(DSN)

    with pytest.raises(ValueError, match="query must not be blank"):
        await store.search_bm25(query, top_k=3, category=None)


@pytest.mark.parametrize("top_k", [-1, 0])
@pytest.mark.asyncio
async def test_search_bm25_rejects_non_positive_top_k(top_k: int) -> None:
    store = PgVectorStore(DSN)

    with pytest.raises(ValueError, match="top_k must be positive"):
        await store.search_bm25("плитка", top_k=top_k, category=None)


def test_bm25_sql_keeps_indexable_ascending_operator_order() -> None:
    select_clause, order_by_clause = BM25_SEARCH_SQL.split("ORDER BY", maxsplit=1)

    assert "-(" in select_clause
    assert "<@> to_bm25query" in select_clause
    assert "<@> to_bm25query" in order_by_clause
    assert "DESC" not in order_by_clause
