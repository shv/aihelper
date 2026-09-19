import asyncio
import json
import sys
from pathlib import Path

from mcp import Client, StdioServerParameters

PROJECT_ROOT = Path(__file__).resolve().parents[1]

TOOL_ARGUMENTS: dict[str, float] = {
    "room_length_m": 4.2,
    "room_width_m": 2.8,
    "tile_length_cm": 60,
    "tile_width_cm": 60,
    "waste_percent": 10,
}


async def main() -> None:
    server = StdioServerParameters(
        command=sys.executable,
        args=["-m", "app.mcp.server"],
        cwd=PROJECT_ROOT,
        encoding="utf-8",
        encoding_error_handler="strict",
    )

    async with Client(
        server,
        raise_exceptions=False,
        read_timeout_seconds=5.0,
        mode="auto",
        input_required_max_rounds=1,
    ) as client:
        tools_result = await client.list_tools()
        call_result = await client.call_tool(
            "calculate_floor_tiles",
            TOOL_ARGUMENTS,
        )

        if call_result.is_error:
            raise RuntimeError(f"MCP tool failed: {call_result.content}")

        if call_result.structured_content is None:
            raise RuntimeError("MCP tool returned no structured content")

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
            "result": call_result.structured_content,
        }

    print(json.dumps(payload, ensure_ascii=False))


if __name__ == "__main__":
    asyncio.run(main())
