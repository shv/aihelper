from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from typing import Any

import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric import rsa
from jwt import PyJWKClient
from jwt.algorithms import RSAAlgorithm

from app.mcp.server import Auth0TokenVerifier

ISSUER_URL = "https://auth.example.test/"
AUDIENCE = "https://api.example.test/mcp"
JWKS_URL = f"{ISSUER_URL}.well-known/jwks.json"
REQUIRED_SCOPE = "repair:use"
KEY_ID = "unit-test-key"
CLIENT_ID = "chatgpt-test-client"
SUBJECT = "auth0|unit-test-user"


@pytest.fixture
def private_key() -> rsa.RSAPrivateKey:
    return rsa.generate_private_key(
        public_exponent=65537,
        key_size=2048,
    )


@pytest.fixture
def jwks(private_key: rsa.RSAPrivateKey) -> dict[str, list[dict[str, Any]]]:
    public_jwk = RSAAlgorithm.to_jwk(private_key.public_key(), as_dict=True)
    public_jwk["kid"] = KEY_ID
    public_jwk["use"] = "sig"
    public_jwk["alg"] = "RS256"
    return {"keys": [public_jwk]}


@pytest.fixture
def mock_jwks_endpoint(
    monkeypatch: pytest.MonkeyPatch,
    jwks: dict[str, list[dict[str, Any]]],
) -> None:
    def fetch_data(_client: PyJWKClient) -> dict[str, list[dict[str, Any]]]:
        return jwks

    monkeypatch.setattr(PyJWKClient, "fetch_data", fetch_data)


@pytest.fixture
def verifier(mock_jwks_endpoint: None) -> Auth0TokenVerifier:
    return Auth0TokenVerifier(
        audience=AUDIENCE,
        jwks_url=JWKS_URL,
        issuer_url=ISSUER_URL,
        required_scope=REQUIRED_SCOPE,
    )


@pytest.fixture
def build_token(
    private_key: rsa.RSAPrivateKey,
) -> Callable[..., str]:
    def factory(
        *,
        omit_claim: str | None = None,
        **overrides: Any,
    ) -> str:
        now = datetime.now(UTC)
        payload: dict[str, Any] = {
            "iss": ISSUER_URL,
            "aud": AUDIENCE,
            "iat": now,
            "exp": now + timedelta(minutes=5),
            "sub": SUBJECT,
            "azp": CLIENT_ID,
            "scope": f"openid profile {REQUIRED_SCOPE}",
        }
        payload.update(overrides)
        if omit_claim is not None:
            del payload[omit_claim]
        return jwt.encode(
            payload=payload,
            key=private_key,
            algorithm="RS256",
            headers={"kid": KEY_ID},
        )

    return factory


@pytest.mark.asyncio
async def test_auth0_token_verifier_accepts_valid_signed_token(
    verifier: Auth0TokenVerifier,
    build_token: Callable[..., str],
) -> None:
    encoded_token = build_token()

    access_token = await verifier.verify_token(encoded_token)

    assert access_token is not None
    assert access_token.token == encoded_token
    assert access_token.client_id == CLIENT_ID
    assert access_token.scopes == ["openid", "profile", REQUIRED_SCOPE]
    assert access_token.expires_at is not None
    assert access_token.resource == AUDIENCE
    assert access_token.subject == SUBJECT
    assert access_token.claims == {"iss": ISSUER_URL}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "claim_overrides",
    [
        {"iss": "https://wrong-issuer.example.test/"},
        {"aud": "https://wrong-audience.example.test/mcp"},
        {"exp": datetime.now(UTC) - timedelta(minutes=1)},
    ],
    ids=["wrong-issuer", "wrong-audience", "expired"],
)
async def test_auth0_token_verifier_rejects_invalid_registered_claims(
    verifier: Auth0TokenVerifier,
    build_token: Callable[..., str],
    claim_overrides: dict[str, Any],
) -> None:
    encoded_token = build_token(**claim_overrides)

    assert await verifier.verify_token(encoded_token) is None


@pytest.mark.asyncio
async def test_auth0_token_verifier_rejects_missing_required_scope(
    verifier: Auth0TokenVerifier,
    build_token: Callable[..., str],
) -> None:
    encoded_token = build_token(scope="openid profile")

    assert await verifier.verify_token(encoded_token) is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "claim_name",
    ["iss", "aud", "exp", "iat", "sub"],
)
async def test_auth0_token_verifier_rejects_missing_required_claim(
    verifier: Auth0TokenVerifier,
    build_token: Callable[..., str],
    claim_name: str,
) -> None:
    encoded_token = build_token(omit_claim=claim_name)

    assert await verifier.verify_token(encoded_token) is None


@pytest.mark.asyncio
async def test_auth0_token_verifier_rejects_token_without_client_id(
    verifier: Auth0TokenVerifier,
    build_token: Callable[..., str],
) -> None:
    encoded_token = build_token(azp=None, client_id=None)

    assert await verifier.verify_token(encoded_token) is None


@pytest.mark.asyncio
async def test_auth0_token_verifier_uses_client_id_when_azp_is_absent(
    verifier: Auth0TokenVerifier,
    build_token: Callable[..., str],
) -> None:
    encoded_token = build_token(
        omit_claim="azp",
        client_id=CLIENT_ID,
    )

    access_token = await verifier.verify_token(encoded_token)

    assert access_token is not None
    assert access_token.client_id == CLIENT_ID


@pytest.mark.asyncio
async def test_auth0_token_verifier_normalizes_audience_list_to_resource(
    verifier: Auth0TokenVerifier,
    build_token: Callable[..., str],
) -> None:
    encoded_token = build_token(
        aud=[AUDIENCE, f"{ISSUER_URL}userinfo"],
    )

    access_token = await verifier.verify_token(encoded_token)

    assert access_token is not None
    assert access_token.resource == AUDIENCE


@pytest.mark.asyncio
async def test_auth0_token_verifier_rejects_malformed_token(
    verifier: Auth0TokenVerifier,
) -> None:
    assert await verifier.verify_token("not-a-jwt") is None


@pytest.mark.asyncio
async def test_auth0_token_verifier_rejects_unknown_signing_key(
    verifier: Auth0TokenVerifier,
    private_key: rsa.RSAPrivateKey,
) -> None:
    now = datetime.now(UTC)
    encoded_token = jwt.encode(
        payload={
            "iss": ISSUER_URL,
            "aud": AUDIENCE,
            "iat": now,
            "exp": now + timedelta(minutes=5),
            "sub": SUBJECT,
            "azp": CLIENT_ID,
            "scope": REQUIRED_SCOPE,
        },
        key=private_key,
        algorithm="RS256",
        headers={"kid": "unknown-key"},
    )

    assert await verifier.verify_token(encoded_token) is None


@pytest.mark.asyncio
async def test_auth0_token_verifier_rejects_invalid_signature(
    verifier: Auth0TokenVerifier,
) -> None:
    untrusted_private_key = rsa.generate_private_key(
        public_exponent=65537,
        key_size=2048,
    )
    now = datetime.now(UTC)
    encoded_token = jwt.encode(
        payload={
            "iss": ISSUER_URL,
            "aud": AUDIENCE,
            "iat": now,
            "exp": now + timedelta(minutes=5),
            "sub": SUBJECT,
            "azp": CLIENT_ID,
            "scope": REQUIRED_SCOPE,
        },
        key=untrusted_private_key,
        algorithm="RS256",
        headers={"kid": KEY_ID},
    )

    assert await verifier.verify_token(encoded_token) is None
