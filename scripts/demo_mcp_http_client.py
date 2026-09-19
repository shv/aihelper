import asyncio
import json
from pathlib import Path

from mcp import Client
from mcp.types import TextResourceContents

PROJECT_ROOT = Path(__file__).resolve().parents[1]

TOOL_ARGUMENTS: dict[str, float] = {
    "room_length_m": 4.2,
    "room_width_m": 2.8,
    "tile_length_cm": 60,
    "tile_width_cm": 60,
    "waste_percent": 10,
}


async def main() -> None:
    async with Client(
        "http://127.0.0.1:8001/mcp",
        raise_exceptions=False,
        read_timeout_seconds=5.0,
        mode="auto",
        input_required_max_rounds=1,
    ) as client:
        tools_result = await client.list_tools(cache_mode="use")
        call_result = await client.call_tool(
            "calculate_floor_tiles",
            TOOL_ARGUMENTS,
        )

        if call_result.is_error:
            raise RuntimeError(f"MCP tool failed: {call_result.content}")

        if call_result.structured_content is None:
            raise RuntimeError("MCP tool returned no structured content")

        resources_result = await client.list_resources(
            cache_mode="use",
        )
        templates_result = await client.list_resource_templates(
            cache_mode="use",
        )
        resource_result = await client.read_resource(
            "repair://knowledge/tile-waterproofing",
            cache_mode="use",
        )

        if len(resource_result.contents) != 1:
            raise RuntimeError("Expected exactly one resource content block")

        resource_content = resource_result.contents[0]

        if not isinstance(resource_content, TextResourceContents):
            raise TypeError("Expected text resource content")

        payload = {
            "server": (
                client.server_info.model_dump(
                    mode="json",
                    by_alias=True,
                    exclude_none=True,
                )
                if client.server_info is not None
                else None
            ),
            "protocol_version": client.protocol_version,
            "tools": [
                tool.model_dump(
                    mode="json",
                    by_alias=True,
                    exclude_none=True,
                )
                for tool in tools_result.tools
            ],
            "resources": [
                resource.model_dump(
                    mode="json",
                    by_alias=True,
                    exclude_none=True,
                )
                for resource in resources_result.resources
            ],
            "resource_templates": [
                template.model_dump(
                    mode="json",
                    by_alias=True,
                    exclude_none=True,
                )
                for template in templates_result.resource_templates
            ],
            "resource": json.loads(resource_content.text),
            "result": call_result.structured_content,
        }

    print(json.dumps(payload, ensure_ascii=False))


if __name__ == "__main__":
    asyncio.run(main())
