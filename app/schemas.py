from enum import StrEnum
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field


class ToolChatResponse(BaseModel):
    answer: str
    tools_used: list[str]


class RiskLevel(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class RepairRisk(BaseModel):
    level: RiskLevel = Field(description="Уровень опасности риска")
    description: str = Field(description="Что может произойти")
    mitigation: str = Field(description="Как снизить или исключить риск")


class RepairAdvice(BaseModel):
    summary: str = Field(description="Краткое резюме ситуации")
    clarifying_questions: list[str] = Field(
        description="Каких исходных данных не хватает"
    )
    recommendations: list[str] = Field(
        description="Рекомендуемые действия в безопасном порядке"
    )
    risks: list[RepairRisk] = Field(description="Обнаруженные риски")
    requires_professional: bool = Field(
        description="Нужно ли привлекать квалифицированного специалиста"
    )


class RagCitation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_id: str = Field(min_length=1, description="ID фрагмента базы знаний")
    quote: str = Field(min_length=1, description="Дословная цитата из текста источника")


class GroundedRepairAdvice(RepairAdvice):
    citations: list[RagCitation] = Field(
        min_length=1,
        description="Дословные цитаты, подтверждающие фактические утверждения ответа",
    )


class ChatRequest(BaseModel):
    message: Annotated[str, Field(min_length=1, max_length=4_000)]


class TokenUsage(BaseModel):
    input_tokens: int
    output_tokens: int
    total_tokens: int


class ChatResponse(BaseModel):
    advice: RepairAdvice
    model: str
    usage: TokenUsage


class RagAnswerStatus(StrEnum):
    ANSWERED = "answered"
    INSUFFICIENT_CONTEXT = "insufficient_context"


class RagChatRequest(ChatRequest):
    category: str | None = None


class RagSource(BaseModel):
    id: str
    title: str
    text: str
    score: float


class RagChatResponse(BaseModel):
    status: RagAnswerStatus
    advice: RepairAdvice
    citations: list[RagCitation]
    model: str | None
    usage: TokenUsage | None
    sources: list[RagSource]
