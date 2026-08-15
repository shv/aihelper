from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse

from app.llm.exceptions import (
    LLMAuthenticationError,
    LLMInvalidResponseError,
    LLMRateLimitError,
    LLMTimeoutError,
    LLMUnavailableError,
)


def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(LLMTimeoutError)
    async def handle_timeout(
        _request: Request,
        _error: LLMTimeoutError,
    ) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            content={"detail": "LLM provider timed out"},
        )

    @app.exception_handler(LLMRateLimitError)
    async def handle_rate_limit(
        _request: Request,
        _error: LLMRateLimitError,
    ) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={"detail": "LLM provider is temporarily overloaded"},
        )

    @app.exception_handler(LLMUnavailableError)
    async def handle_unavailable(
        _request: Request,
        _error: LLMUnavailableError,
    ) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={"detail": "LLM provider is unavailable"},
        )

    @app.exception_handler(LLMInvalidResponseError)
    async def handle_invalid_response(
        _request: Request,
        _error: LLMInvalidResponseError,
    ) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_502_BAD_GATEWAY,
            content={"detail": "LLM provider returned an invalid response"},
        )

    @app.exception_handler(LLMAuthenticationError)
    async def handle_authentication(
        _request: Request,
        _error: LLMAuthenticationError,
    ) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"detail": "LLM provider is misconfigured"},
        )
