from typing import Any

import pytest
from mcp import Client

from app.mcp.server import MCP_SERVER_NAME, mcp

TOOL_ARGUMENTS: dict[str, float] = {
    "room_length_m": 4.2,
    "room_width_m": 2.8,
    "tile_length_cm": 60,
    "tile_width_cm": 60,
    "waste_percent": 10,
}


@pytest.mark.asyncio
async def test_mcp_server_publishes_tile_tool_contract() -> None:
    async with Client(
        mcp,
        raise_exceptions=True,
        read_timeout_seconds=5.0,
    ) as client:
        result = await client.list_tools()
        server_info = client.server_info

    assert server_info is not None
    assert server_info.name == MCP_SERVER_NAME
    assert server_info.version == "0.1.0"
    assert len(result.tools) == 1

    tool = result.tools[0]
    assert tool.name == "calculate_floor_tiles"
    assert tool.title == "Расчёт напольной плитки"
    assert tool.annotations is not None
    assert tool.annotations.read_only_hint is True
    assert tool.annotations.open_world_hint is False
    assert tool.output_schema is not None
    assert tool.output_schema["type"] == "object"
    assert tool.output_schema["required"] == [
        "room_area_m2",
        "required_area_m2",
        "tile_area_m2",
        "tile_count",
        "actual_coverage_m2",
        "waste_percent",
    ]


@pytest.mark.asyncio
async def test_mcp_tool_input_schema_preserves_domain_constraints() -> None:
    async with Client(
        mcp,
        raise_exceptions=True,
        read_timeout_seconds=5.0,
    ) as client:
        result = await client.list_tools()

    properties: dict[str, dict[str, Any]] = result.tools[0].input_schema["properties"]

    assert properties["room_length_m"] == {
        "description": "Длина комнаты в метрах",
        "exclusiveMinimum": 0,
        "maximum": 100,
        "title": "Room Length M",
        "type": "number",
    }
    assert properties["waste_percent"] == {
        "description": "Запас плитки на подрезку в процентах",
        "maximum": 50,
        "minimum": 0,
        "title": "Waste Percent",
        "type": "number",
    }


@pytest.mark.asyncio
async def test_mcp_tool_returns_structured_tile_calculation() -> None:
    async with Client(
        mcp,
        raise_exceptions=True,
        read_timeout_seconds=5.0,
    ) as client:
        result = await client.call_tool(
            "calculate_floor_tiles",
            TOOL_ARGUMENTS,
        )

    assert result.is_error is False
    assert result.structured_content == {
        "room_area_m2": 11.76,
        "required_area_m2": 12.94,
        "tile_area_m2": 0.36,
        "tile_count": 36,
        "actual_coverage_m2": 12.96,
        "waste_percent": 10.0,
    }


@pytest.mark.asyncio
async def test_mcp_tool_reports_invalid_arguments_as_tool_error() -> None:
    async with Client(
        mcp,
        raise_exceptions=False,
        read_timeout_seconds=5.0,
    ) as client:
        result = await client.call_tool(
            "calculate_floor_tiles",
            {
                **TOOL_ARGUMENTS,
                "room_length_m": 0,
            },
        )

    assert result.is_error is True
    assert result.structured_content is None
