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
    LLMInvalidResponseError,
    LLMRateLimitError,
    LLMTimeoutError,
    LLMUnavailableError,
)
from app.schemas import RepairAdvice, TokenUsage

from .base import RepairAdviceResult
from .prompts import REPAIR_ASSISTANT_INSTRUCTIONS


class OpenAIRepairAdviceProvider:
    def __init__(self, client: AsyncOpenAI, model: str) -> None:
        self._client = client
        self._model = model

    async def get_repair_advice(self, message: str) -> RepairAdviceResult:
        try:
            response = await self._client.responses.parse(
                model=self._model,
                reasoning={"effort": "low"},
                instructions=REPAIR_ASSISTANT_INSTRUCTIONS,
                input=message,
                text_format=RepairAdvice,
            )
        except APITimeoutError as error:
            raise LLMTimeoutError("OpenAI request timed out") from error
        except RateLimitError as error:
            raise LLMRateLimitError("OpenAI rate limit exceeded") from error
        except AuthenticationError as error:
            raise LLMAuthenticationError("OpenAI authentication failed") from error
        except APIConnectionError as error:
            raise LLMUnavailableError("Cannot connect to OpenAI") from error
        except APIError as error:
            raise LLMUnavailableError("OpenAI API error") from error

        if response.output_parsed is None:
            raise LLMInvalidResponseError("OpenAI did not return structured output")

        if response.usage is None:
            raise LLMInvalidResponseError("OpenAI did not return token usage")

        return RepairAdviceResult(
            advice=response.output_parsed,
            model=response.model,
            usage=TokenUsage(
                input_tokens=response.usage.input_tokens,
                output_tokens=response.usage.output_tokens,
                total_tokens=response.usage.total_tokens,
            ),
        )
