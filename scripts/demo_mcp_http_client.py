"""
Терминал 1. Запуск сервера
```
export MCP_ISSUER_URL='...'
export MCP_RESOURCE_SERVER_URL='...'
export MCP_REQUIRED_SCOPE='...'

poetry run python -m scripts.run_mcp_http
```


Терминал 2. Последовательность получения токена (нужно в Auth0 разрешить scope)
```
export AUTH0_DOMAIN='dev-0af4kgvu3aggrajc.us.auth0.com'
export AUTH0_CLIENT_ID='твой-client-id'

read -s "AUTH0_CLIENT_SECRET?Auth0 Client Secret: "
echo
```

```
export MCP_ISSUER_URL='...'
export MCP_RESOURCE_SERVER_URL='...'
export MCP_REQUIRED_SCOPE='...'
```

```
export MCP_ACCESS_TOKEN="$(
  curl -fsS \
    -X POST "https://${AUTH0_DOMAIN}/oauth/token" \
    --data-urlencode 'grant_type=client_credentials' \
    --data-urlencode "client_id=${AUTH0_CLIENT_ID}" \
    --data-urlencode "client_secret=${AUTH0_CLIENT_SECRET}" \
    --data-urlencode audience=${MCP_ISSUER_URL}" \
    --data-urlencode scope=${MCP_REQUIRED_SCOPE}" \
  | poetry run python -c \
    'import json, sys; print(json.load(sys.stdin)["access_token"])'
)"
```

```
poetry run python -m scripts.run_mcp_http
```

```
poetry run python -m scripts.demo_mcp_http_client \
  http://127.0.0.1:8001/mcp \
  | json_pp
```
"""

import asyncio
import json
import os
import sys
from pathlib import Path

import httpx2
from mcp import Client
from mcp.client.streamable_http import streamable_http_client
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
    url = (sys.argv[1] if len(sys.argv) > 1 else None) or "http://127.0.0.1:8001/mcp"
    token = os.environ["MCP_ACCESS_TOKEN"]
    async with (
        httpx2.AsyncClient(
            headers={
                "Authorization": f"Bearer {token}",
            },
        ) as http_client,
        Client(
            streamable_http_client(
                url,
                http_client=http_client,
            ),
            raise_exceptions=False,
            read_timeout_seconds=5.0,
            mode="auto",
            input_required_max_rounds=1,
        ) as client,
    ):
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
