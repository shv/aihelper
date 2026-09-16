import pytest
from pydantic import ValidationError

from app.tools.tile import TileCalculationInput, calculate_floor_tiles


def test_calculate_floor_tiles_rounds_up_to_whole_tiles() -> None:
    arguments = TileCalculationInput(
        room_length_m=4.2,
        room_width_m=2.8,
        tile_length_cm=60,
        tile_width_cm=60,
        waste_percent=10,
    )

    result = calculate_floor_tiles(arguments)

    assert result == {
        "room_area_m2": 11.76,
        "required_area_m2": 12.94,
        "tile_area_m2": 0.36,
        "tile_count": 36,
        "actual_coverage_m2": 12.96,
        "waste_percent": 10,
    }


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("room_length_m", 0),
        ("room_length_m", 101),
        ("room_width_m", 0),
        ("room_width_m", 101),
        ("tile_length_cm", 0),
        ("tile_length_cm", 501),
        ("tile_width_cm", 0),
        ("tile_width_cm", 501),
        ("waste_percent", -1),
        ("waste_percent", 51),
    ],
)
def test_tile_calculation_input_rejects_out_of_range_values(
    field: str,
    value: float,
) -> None:
    payload = {
        "room_length_m": 4.2,
        "room_width_m": 2.8,
        "tile_length_cm": 60,
        "tile_width_cm": 60,
        "waste_percent": 10,
    }
    payload[field] = value

    with pytest.raises(ValidationError):
        TileCalculationInput.model_validate(payload)


def test_tile_calculation_input_rejects_unknown_fields() -> None:
    payload = {
        "room_length_m": 4.2,
        "room_width_m": 2.8,
        "tile_length_cm": 60,
        "tile_width_cm": 60,
        "waste_percent": 10,
        "unknown": "value",
    }

    with pytest.raises(ValidationError):
        TileCalculationInput.model_validate(payload)
