import json
import subprocess
import sys
from typing import Any

from scripts.demo_mcp_client import PROJECT_ROOT


def test_demo_mcp_client_lists_and_calls_tool_over_stdio() -> None:
    completed = subprocess.run(
        [sys.executable, "-m", "scripts.demo_mcp_client"],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=10.0,
    )

    assert completed.returncode == 0, completed.stderr

    payload: dict[str, Any] = json.loads(completed.stdout)

    assert payload["server"] == {
        "name": "aihelper-repair-tools",
        "version": "0.1.0",
    }
    assert payload["protocol_version"] == "2026-07-28"
    assert len(payload["tools"]) == 1
    assert payload["tools"][0]["name"] == "calculate_floor_tiles"
    assert payload["result"] == {
        "room_area_m2": 11.76,
        "required_area_m2": 12.94,
        "tile_area_m2": 0.36,
        "tile_count": 36,
        "actual_coverage_m2": 12.96,
        "waste_percent": 10.0,
    }
