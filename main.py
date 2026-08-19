import json
import math
import os
from collections.abc import AsyncIterator
from functools import lru_cache
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.responses import StreamingResponse
from openai import AsyncOpenAI
from openai.types.responses import FunctionToolParam, ResponseInputParam

from app.http_errors import register_exception_handlers
from app.llm.base import GroundedRepairAdviceProvider, RepairAdviceProvider
from app.llm.openai import OpenAIRepairAdviceProvider
from app.llm.prompts import REPAIR_ASSISTANT_INSTRUCTIONS
from app.rag.embeddings import EmbeddingProvider, OpenAIEmbeddingProvider
from app.rag.service import RagService, SearchStore
from app.rag.store import PgVectorStore
from app.schemas import (
    ChatRequest,
    ChatResponse,
    RagChatRequest,
    RagChatResponse,
    RagSource,
    TileCalculationInput,
    ToolChatResponse,
)

app = FastAPI(
    title="AI Helper",
    description="Simple FastAPI application",
    version="0.1.0",
)

register_exception_handlers(app)


@app.get("/")
async def root() -> dict[str, str]:
    return {"message": "AI Helper API is running"}


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


EMBEDDING_MODEL = "text-embedding-3-small"
MODEL = "gpt-5.6-luna"  # "gpt-5.6" - дороже
RAG_MIN_SCORE = 0.45
RAG_TOP_K = 3


@lru_cache
def get_openai_client() -> AsyncOpenAI:
    return AsyncOpenAI(
        timeout=20.0,
        max_retries=2,
    )


OpenAIClientDependency = Annotated[
    AsyncOpenAI,
    Depends(get_openai_client),
]


def get_repair_advice_provider(
    client: OpenAIClientDependency,
) -> RepairAdviceProvider:
    return OpenAIRepairAdviceProvider(
        client=client,
        model=MODEL,
    )


CALCULATE_FLOOR_TILES_TOOL = FunctionToolParam(
    type="function",
    name="calculate_floor_tiles",
    description=(
        "Рассчитать количество напольной плитки для прямоугольной комнаты. "
        "Используй инструмент, когда пользователь указал размеры комнаты, "
        "размер плитки и запас."
    ),
    parameters=TileCalculationInput.model_json_schema(),
    strict=True,
)


@app.post("/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    provider: Annotated[RepairAdviceProvider, Depends(get_repair_advice_provider)],
) -> ChatResponse:
    result = await provider.get_repair_advice(request.message)

    return ChatResponse(
        advice=result.advice,
        model=result.model,
        usage=result.usage,
    )


# Streaming
def make_sse_event(event: str, data: dict[str, object]) -> str:
    payload = json.dumps(data, ensure_ascii=False)
    return f"event: {event}\ndata: {payload}\n\n"


async def generate_chat_stream(message: str, client: AsyncOpenAI) -> AsyncIterator[str]:
    stream = await client.responses.create(
        model=MODEL,
        reasoning={"effort": "low"},
        instructions=REPAIR_ASSISTANT_INSTRUCTIONS,
        input=message,
        stream=True,
    )
    async for event in stream:
        if event.type == "response.output_text.delta":
            yield make_sse_event(event="token", data={"delta": event.delta})
        elif event.type == "response.completed":
            usage = event.response.usage
            yield make_sse_event(
                event="done",
                data={
                    "model": event.response.model,
                    "usage": (
                        {
                            "input_tokens": usage.input_tokens,
                            "output_tokens": usage.output_tokens,
                            "total_tokens": usage.total_tokens,
                        }
                        if usage is not None
                        else None
                    ),
                },
            )


@app.post("/chat/stream")
async def chat_stream(
    request: ChatRequest, client: OpenAIClientDependency
) -> StreamingResponse:
    return StreamingResponse(
        generate_chat_stream(request.message, client),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


def calculate_floor_tiles(arguments: TileCalculationInput) -> dict[str, object]:
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


@app.post("/chat/tools", response_model=ToolChatResponse)
async def chat_with_tools(
    request: ChatRequest, client: OpenAIClientDependency
) -> ToolChatResponse:
    first_response = await client.responses.create(
        model=MODEL,
        reasoning={"effort": "low"},
        instructions=REPAIR_ASSISTANT_INSTRUCTIONS,
        input=request.message,
        tools=[CALCULATE_FLOOR_TILES_TOOL],
        tool_choice="auto",
    )

    tool_outputs: ResponseInputParam = []
    tools_used: list[str] = []

    for item in first_response.output:
        if item.type != "function_call":
            continue

        if item.name != "calculate_floor_tiles":
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=f"Unknown tool: {item.name}",
            )

        print(f"Item arguments: {item.arguments}")

        try:
            arguments = TileCalculationInput.model_validate_json(item.arguments)
        except ValueError as error:
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail="LLM returned invalid tool arguments",
            ) from error

        print(f"Arguments: {arguments}")

        result = calculate_floor_tiles(arguments)

        tool_outputs.append(
            {
                "type": "function_call_output",
                "call_id": item.call_id,
                "output": json.dumps(result, ensure_ascii=False),
            }
        )
        tools_used.append(item.name)

    if not tool_outputs:
        return ToolChatResponse(
            answer=first_response.output_text,
            tools_used=[],
        )

    final_response = await client.responses.create(
        model=MODEL,
        reasoning={"effort": "low"},
        instructions=REPAIR_ASSISTANT_INSTRUCTIONS,
        previous_response_id=first_response.id,
        tools=[CALCULATE_FLOOR_TILES_TOOL],
        input=tool_outputs,
    )

    return ToolChatResponse(
        answer=final_response.output_text,
        tools_used=tools_used,
    )


def get_embedding_provider(client: OpenAIClientDependency) -> EmbeddingProvider:
    return OpenAIEmbeddingProvider(client=client, model=EMBEDDING_MODEL)


def get_grounded_repair_advice_provider(
    client: OpenAIClientDependency,
) -> GroundedRepairAdviceProvider:
    return OpenAIRepairAdviceProvider(client=client, model=MODEL)


def get_search_store() -> SearchStore:
    return PgVectorStore(dsn=os.environ["DATABASE_URL"])


EmbeddingProviderDependency = Annotated[
    EmbeddingProvider, Depends(get_embedding_provider)
]

SearchStoreDependency = Annotated[SearchStore, Depends(get_search_store)]

GroundedProviderDependency = Annotated[
    GroundedRepairAdviceProvider, Depends(get_grounded_repair_advice_provider)
]


def get_rag_service(
    embedding_provider: EmbeddingProviderDependency,
    search_store: SearchStoreDependency,
    advice_provider: GroundedProviderDependency,
) -> RagService:
    return RagService(
        embedding_provider=embedding_provider,
        search_store=search_store,
        advice_provider=advice_provider,
        embedding_model=EMBEDDING_MODEL,
        top_k=RAG_TOP_K,
        min_score=RAG_MIN_SCORE,
    )


@app.post("/chat/rag", response_model=RagChatResponse)
async def chat_rag(
    request: RagChatRequest, service: Annotated[RagService, Depends(get_rag_service)]
) -> RagChatResponse:
    result = await service.get_repair_advice(request.message, category=request.category)

    return RagChatResponse(
        status=result.status,
        advice=result.advice,
        model=result.model,
        usage=result.usage,
        sources=[
            RagSource(
                id=source.chunk.id,
                title=source.chunk.title,
                text=source.chunk.text,
                score=source.score,
            )
            for source in result.sources
        ],
    )
