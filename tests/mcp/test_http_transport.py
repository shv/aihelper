import json

import httpx2
import pytest
from mcp import Client
from mcp.client.streamable_http import streamable_http_client
from mcp.types import TextResourceContents
from starlette.applications import Starlette

from app.mcp.server import mcp

MCP_URL = "http://127.0.0.1:8001/mcp"
MCP_BEARER_TOKEN = "unit-test-mcp-bearer-token"


def create_http_app() -> Starlette:
    return mcp.streamable_http_app(
        host="127.0.0.1",
        streamable_http_path="/mcp",
        json_response=False,
        stateless_http=False,
        max_request_body_size=4 * 1024 * 1024,
        session_idle_timeout=1800.0,
        max_sessions=100,
    )


@pytest.mark.asyncio
async def test_mcp_tools_and_resource_over_streamable_http() -> None:
    app = create_http_app()

    async with (
        app.router.lifespan_context(app),
        httpx2.AsyncClient(
            transport=httpx2.ASGITransport(app=app),
            base_url="http://127.0.0.1:8001",
            headers={"Authorization": f"Bearer {MCP_BEARER_TOKEN}"},
        ) as http_client,
        Client(
            streamable_http_client(MCP_URL, http_client=http_client),
            mode="2026-07-28",
            raise_exceptions=True,
            read_timeout_seconds=5.0,
            input_required_max_rounds=1,
        ) as client,
    ):
        tools = await client.list_tools(cache_mode="use")
        templates = await client.list_resource_templates(cache_mode="use")
        calculation = await client.call_tool(
            "calculate_floor_tiles",
            {
                "room_length_m": 4.2,
                "room_width_m": 2.8,
                "tile_length_cm": 60,
                "tile_width_cm": 60,
                "waste_percent": 10,
            },
        )
        resource = await client.read_resource(
            "repair://knowledge/tile-waterproofing",
            cache_mode="use",
        )
        protocol_version = client.protocol_version

    assert protocol_version == "2026-07-28"
    assert [tool.name for tool in tools.tools] == ["calculate_floor_tiles"]
    assert [template.uri_template for template in templates.resource_templates] == [
        "repair://knowledge/{chunk_id}"
    ]
    assert calculation.is_error is False
    assert calculation.structured_content is not None
    assert calculation.structured_content["tile_count"] == 36
    assert len(resource.contents) == 1
    content = resource.contents[0]
    assert isinstance(content, TextResourceContents)
    assert json.loads(content.text)["id"] == "tile-waterproofing"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "authorization",
    [
        None,
        "Bearer wrong-token",
    ],
    ids=["missing-token", "wrong-token"],
)
async def test_mcp_http_rejects_unauthenticated_requests(
    authorization: str | None,
) -> None:
    app = create_http_app()
    headers = {"Authorization": authorization} if authorization is not None else None

    async with (
        app.router.lifespan_context(app),
        httpx2.AsyncClient(
            transport=httpx2.ASGITransport(app=app),
            base_url="http://127.0.0.1:8001",
        ) as http_client,
    ):
        response = await http_client.post(
            "/mcp",
            headers=headers,
            json={
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2026-07-28",
                    "capabilities": {},
                    "clientInfo": {
                        "name": "test-client",
                        "version": "0.1.0",
                    },
                },
            },
        )

    assert response.status_code == 401
    assert response.json() == {
        "error": "invalid_token",
        "error_description": "Authentication required",
    }
    assert response.headers["www-authenticate"].startswith("Bearer ")
