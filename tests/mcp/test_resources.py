import json

import pytest
from mcp import Client
from mcp.shared.exceptions import MCPError
from mcp.types import TextResourceContents

from app.mcp.server import mcp
from app.rag.demo_documents import CHUNKS_BY_ID


@pytest.mark.asyncio
async def test_knowledge_resource_is_template_not_fixed_resource() -> None:
    async with Client(
        mcp,
        raise_exceptions=True,
        read_timeout_seconds=5.0,
    ) as client:
        resources = await client.list_resources(cache_mode="use")
        templates = await client.list_resource_templates(cache_mode="use")

    assert resources.resources == []
    assert len(templates.resource_templates) == 1
    template = templates.resource_templates[0]
    assert template.uri_template == "repair://knowledge/{chunk_id}"
    assert template.name == "repair_knowledge_chunk"
    assert template.mime_type == "application/json"


@pytest.mark.asyncio
async def test_read_knowledge_resource_returns_existing_chunk() -> None:
    chunk = CHUNKS_BY_ID["tile-waterproofing"]
    uri = f"repair://knowledge/{chunk.id}"

    async with Client(
        mcp,
        raise_exceptions=True,
        read_timeout_seconds=5.0,
    ) as client:
        result = await client.read_resource(uri, cache_mode="use")

    assert len(result.contents) == 1
    content = result.contents[0]
    assert isinstance(content, TextResourceContents)
    assert str(content.uri) == uri
    assert content.mime_type == "application/json"
    assert json.loads(content.text) == {
        "id": chunk.id,
        "title": chunk.title,
        "text": chunk.text,
        "metadata": chunk.metadata,
    }


@pytest.mark.asyncio
async def test_read_knowledge_resource_rejects_unknown_chunk() -> None:
    async with Client(
        mcp,
        raise_exceptions=True,
        read_timeout_seconds=5.0,
    ) as client:
        with pytest.raises(MCPError, match="Unknown knowledge chunk: missing"):
            await client.read_resource(
                "repair://knowledge/missing",
                cache_mode="use",
            )
