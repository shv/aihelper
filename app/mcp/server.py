import asyncio
import json
import os

import jwt
from jwt import PyJWTError
from mcp.server import MCPServer
from mcp.server.auth.provider import AccessToken, TokenVerifier
from mcp.server.auth.settings import AuthSettings
from mcp.server.mcpserver.exceptions import ResourceError
from mcp.types import ToolAnnotations
from pydantic import AnyHttpUrl

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

MCP_ISSUER_URL = os.environ["MCP_ISSUER_URL"]
MCP_JWKS_URL = f"{MCP_ISSUER_URL}.well-known/jwks.json"
MCP_RESOURCE_SERVER_URL = os.environ["MCP_RESOURCE_SERVER_URL"]
MCP_REQUIRED_SCOPE = os.environ["MCP_REQUIRED_SCOPE"]
MCP_AUDIENCE = MCP_RESOURCE_SERVER_URL


class Auth0TokenVerifier(TokenVerifier):
    def __init__(
        self,
        audience: str,
        jwks_url: str,
        issuer_url: str,
        required_scope: str,
    ) -> None:
        self._audience = audience
        self._jwks_url = jwks_url
        self._required_scope = required_scope
        self._issuer_url = issuer_url
        self._jwks_client = jwt.PyJWKClient(self._jwks_url)

        super().__init__()

    async def verify_token(self, token: str) -> AccessToken | None:
        """Verify token using JWT."""
        try:
            signing_key = await asyncio.to_thread(
                self._jwks_client.get_signing_key_from_jwt,
                token,
            )

            payload = jwt.decode(
                jwt=token,
                key=signing_key.key,
                algorithms=["RS256"],
                issuer=self._issuer_url,
                audience=self._audience,
                options={
                    "require": [
                        "iss",
                        "aud",
                        "exp",
                        "iat",
                        "sub",
                    ],
                },
            )
        except PyJWTError:
            print("PyJWTError error")
            return None

        scopes = payload.get("scope", "").split()
        if self._required_scope not in set(scopes):
            print(f"No scope {self._required_scope} in scopes {scopes}")
            return None

        client_id = payload.get("azp") or payload.get("client_id")
        if not client_id:
            print("Client_id not found")
            return None

        return AccessToken(
            token=token,
            client_id=str(client_id),
            scopes=scopes,
            expires_at=payload.get("exp"),
            resource=self._audience,
            subject=payload.get("sub"),
            claims={"iss": payload.get("iss")},
        )


auth_settings = AuthSettings(
    issuer_url=AnyHttpUrl(MCP_ISSUER_URL),
    resource_server_url=AnyHttpUrl(MCP_RESOURCE_SERVER_URL),
    required_scopes=[MCP_REQUIRED_SCOPE],
    validate_token_resource=True,
    identity_assertion_enabled=False,
)


mcp = MCPServer(
    name=MCP_SERVER_NAME,
    version=MCP_SERVER_VERSION,
    debug=False,
    log_level="INFO",
    auth=auth_settings,
    token_verifier=Auth0TokenVerifier(
        audience=MCP_AUDIENCE,
        jwks_url=MCP_JWKS_URL,
        issuer_url=MCP_ISSUER_URL,
        required_scope=MCP_REQUIRED_SCOPE,
    ),
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
