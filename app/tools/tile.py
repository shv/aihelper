import math
from typing import Annotated, TypedDict

from pydantic import BaseModel, ConfigDict, Field

RoomLengthM = Annotated[
    float,
    Field(gt=0, le=100, description="Длина комнаты в метрах"),
]
RoomWidthM = Annotated[
    float,
    Field(gt=0, le=100, description="Ширина комнаты в метрах"),
]
TileLengthCm = Annotated[
    float,
    Field(gt=0, le=500, description="Длина плитки в сантиметрах"),
]
TileWidthCm = Annotated[
    float,
    Field(gt=0, le=500, description="Ширина плитки в сантиметрах"),
]
WastePercent = Annotated[
    float,
    Field(ge=0, le=50, description="Запас плитки на подрезку в процентах"),
]


class TileCalculationInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    room_length_m: RoomLengthM
    room_width_m: RoomWidthM
    tile_length_cm: TileLengthCm
    tile_width_cm: TileWidthCm
    waste_percent: WastePercent


class TileCalculationResult(TypedDict):
    room_area_m2: float
    required_area_m2: float
    tile_area_m2: float
    tile_count: int
    actual_coverage_m2: float
    waste_percent: float


def calculate_floor_tiles(arguments: TileCalculationInput) -> TileCalculationResult:
    room_area_m2 = arguments.room_length_m * arguments.room_width_m
    tile_area_m2 = arguments.tile_length_cm * arguments.tile_width_cm / 10000
    required_area_m2 = room_area_m2 * (1 + arguments.waste_percent / 100)
    tile_count = math.ceil(required_area_m2 / tile_area_m2)
    actual_coverage_m2 = tile_count * tile_area_m2
    return {
        "room_area_m2": round(room_area_m2, 2),
        "required_area_m2": round(required_area_m2, 2),
        "tile_area_m2": round(tile_area_m2, 4),
        "tile_count": tile_count,
        "actual_coverage_m2": round(actual_coverage_m2, 2),
        "waste_percent": arguments.waste_percent,
    }
