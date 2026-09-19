import json

from mcp.server import MCPServer
from mcp.server.mcpserver.exceptions import ResourceError
from mcp.types import ToolAnnotations

from app.rag.demo_documents import CHUNKS_BY_ID
from app.tools.tile import (
    RoomLengthM,
    RoomWidthM,
    TileCalculationInput,
    TileCalculationResult,
    TileLengthCm,
    TileWidthCm,
    WastePercent,
    calculate_floor_tiles,
)

MCP_SERVER_NAME = "aihelper-repair-tools"
MCP_SERVER_VERSION = "0.1.0"

mcp = MCPServer(
    name=MCP_SERVER_NAME,
    version=MCP_SERVER_VERSION,
    debug=False,
    log_level="INFO",
    warn_on_duplicate_resources=True,
    warn_on_duplicate_tools=True,
    warn_on_duplicate_prompts=True,
)


@mcp.tool(
    name="calculate_floor_tiles",
    title="Расчёт напольной плитки",
    description=(
        "Рассчитать количество напольной плитки для прямоугольной комнаты "
        "с учётом запаса на подрезку."
    ),
    annotations=ToolAnnotations(
        read_only_hint=True,
        open_world_hint=False,
    ),
    structured_output=True,
)
def calculate_floor_tiles_tool(
    room_length_m: RoomLengthM,
    room_width_m: RoomWidthM,
    tile_length_cm: TileLengthCm,
    tile_width_cm: TileWidthCm,
    waste_percent: WastePercent,
) -> TileCalculationResult:
    arguments = TileCalculationInput(
        room_length_m=room_length_m,
        room_width_m=room_width_m,
        tile_length_cm=tile_length_cm,
        tile_width_cm=tile_width_cm,
        waste_percent=waste_percent,
    )
    return calculate_floor_tiles(arguments)


@mcp.resource(
    uri="repair://knowledge/{chunk_id}",
    name="repair_knowledge_chunk",
    title="Фрагмент ремонтной базы знаний",
    description="Получить фрагмент ремонтной базы знаний по его идентификатору.",
    mime_type="application/json",
)
def read_repair_knowledge_chunk(chunk_id: str) -> str:
    chunk = CHUNKS_BY_ID.get(chunk_id)

    if chunk is None:
        raise ResourceError(f"Unknown knowledge chunk: {chunk_id}")

    return json.dumps(
        {
            "id": chunk.id,
            "title": chunk.title,
            "text": chunk.text,
            "metadata": chunk.metadata,
        },
        ensure_ascii=False,
    )


if __name__ == "__main__":
    mcp.run("stdio")
