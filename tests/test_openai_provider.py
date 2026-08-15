from types import SimpleNamespace
from typing import cast
from unittest.mock import AsyncMock, MagicMock

import httpx2
import pytest
from openai import (
    APIConnectionError,
    APIError,
    APITimeoutError,
    AsyncOpenAI,
    AuthenticationError,
    RateLimitError,
)

from app.llm.exceptions import (
    LLMAuthenticationError,
    LLMError,
    LLMInvalidResponseError,
    LLMRateLimitError,
    LLMTimeoutError,
    LLMUnavailableError,
)
from app.llm.openai import OpenAIRepairAdviceProvider
from app.llm.prompts import REPAIR_ASSISTANT_INSTRUCTIONS
from app.schemas import RepairAdvice


def make_client(parse_mock: AsyncMock) -> AsyncOpenAI:
    client_mock = MagicMock(spec=AsyncOpenAI)
    client_mock.responses.parse = parse_mock

    return cast(AsyncOpenAI, client_mock)


@pytest.mark.asyncio
async def test_provider_returns_repair_advice() -> None:
    advice = RepairAdvice(
        summary="Тест",
        clarifying_questions=[],
        recommendations=["Проверить размеры"],
        risks=[],
        requires_professional=False,
    )

    raw_response = SimpleNamespace(
        output_parsed=advice,
        model="fake-model",
        usage=SimpleNamespace(
            input_tokens=10,
            output_tokens=20,
            total_tokens=30,
        ),
    )

    parse_mock = AsyncMock(return_value=raw_response)

    provider = OpenAIRepairAdviceProvider(
        client=make_client(parse_mock),
        model="fake-model",
    )

    result = await provider.get_repair_advice("Тестовый вопрос")

    assert result.advice == advice
    assert result.model == "fake-model"
    assert result.usage.input_tokens == 10
    assert result.usage.output_tokens == 20
    assert result.usage.total_tokens == 30

    parse_mock.assert_awaited_once_with(
        model="fake-model",
        reasoning={"effort": "low"},
        instructions=REPAIR_ASSISTANT_INSTRUCTIONS,
        input="Тестовый вопрос",
        text_format=RepairAdvice,
    )


REQUEST = httpx2.Request(
    "POST",
    "https://api.openai.com/v1/responses",
)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("sdk_error", "expected_error"),
    [
        (
            APITimeoutError(REQUEST),
            LLMTimeoutError,
        ),
        (
            APIConnectionError(request=REQUEST),
            LLMUnavailableError,
        ),
        (
            RateLimitError(
                "Rate limit",
                response=httpx2.Response(
                    429,
                    request=REQUEST,
                ),
                body=None,
            ),
            LLMRateLimitError,
        ),
        (
            AuthenticationError(
                "Invalid key",
                response=httpx2.Response(
                    401,
                    request=REQUEST,
                ),
                body=None,
            ),
            LLMAuthenticationError,
        ),
        (
            APIError(
                "Unknown OpenAI error",
                REQUEST,
                body=None,
            ),
            LLMUnavailableError,
        ),
    ],
)
async def test_provider_maps_openai_errors(
    sdk_error: APIError,
    expected_error: type[LLMError],
) -> None:
    parse_mock = AsyncMock(side_effect=sdk_error)

    provider = OpenAIRepairAdviceProvider(
        client=make_client(parse_mock),
        model="fake-model",
    )

    with pytest.raises(expected_error) as captured:
        await provider.get_repair_advice("Тестовый вопрос")

    assert captured.value.__cause__ is sdk_error


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("output_parsed", "usage"),
    [
        (
            None,
            SimpleNamespace(
                input_tokens=10,
                output_tokens=20,
                total_tokens=30,
            ),
        ),
        (
            RepairAdvice(
                summary="Тест",
                clarifying_questions=[],
                recommendations=[],
                risks=[],
                requires_professional=False,
            ),
            None,
        ),
    ],
)
async def test_provider_rejects_incomplete_response(
    output_parsed: RepairAdvice | None,
    usage: object | None,
) -> None:
    parse_mock = AsyncMock(
        return_value=SimpleNamespace(
            output_parsed=output_parsed,
            model="fake-model",
            usage=usage,
        )
    )

    provider = OpenAIRepairAdviceProvider(
        client=make_client(parse_mock),
        model="fake-model",
    )

    with pytest.raises(LLMInvalidResponseError):
        await provider.get_repair_advice("Тестовый вопрос")
