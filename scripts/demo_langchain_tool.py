import json

from langchain_core.tools import StructuredTool

from app.tools.tile import (
    TileCalculationInput,
    TileCalculationResult,
    calculate_floor_tiles,
)


def calculate_floor_tiles_adapter(
    room_length_m: float,
    room_width_m: float,
    tile_length_cm: float,
    tile_width_cm: float,
    waste_percent: float,
) -> TileCalculationResult:
    arguments = TileCalculationInput(
        room_length_m=room_length_m,
        room_width_m=room_width_m,
        tile_length_cm=tile_length_cm,
        tile_width_cm=tile_width_cm,
        waste_percent=waste_percent,
    )

    return calculate_floor_tiles(arguments)


CALCULATE_FLOOR_TILES_LANGCHAIN_TOOL = StructuredTool.from_function(
    func=calculate_floor_tiles_adapter,
    name="calculate_floor_tiles",
    description=(
        "Рассчитать количество напольной плитки для прямоугольной комнаты. "
        "Используй инструмент, когда пользователь указал размеры комнаты, "
        "размер плитки и запас."
    ),
    return_direct=False,
    args_schema=TileCalculationInput,
    infer_schema=False,
    response_format="content",
    parse_docstring=False,
    error_on_invalid_docstring=False,
)


def main() -> None:
    result = CALCULATE_FLOOR_TILES_LANGCHAIN_TOOL.invoke(
        {
            "room_length_m": 4.2,
            "room_width_m": 2.8,
            "tile_length_cm": 60,
            "tile_width_cm": 60,
            "waste_percent": 10,
        }
    )

    payload = {
        "name": CALCULATE_FLOOR_TILES_LANGCHAIN_TOOL.name,
        "description": CALCULATE_FLOOR_TILES_LANGCHAIN_TOOL.description,
        "input_schema": TileCalculationInput.model_json_schema(),
        "result": result,
    }

    print(json.dumps(payload, ensure_ascii=False))


if __name__ == "__main__":
    main()
