from dataclasses import dataclass
from typing import Protocol

from app.schemas import RepairAdvice, TokenUsage


@dataclass(frozen=True, slots=True)
class RepairAdviceResult:
    advice: RepairAdvice
    model: str
    usage: TokenUsage


class RepairAdviceProvider(Protocol):
    async def get_repair_advice(self, message: str) -> RepairAdviceResult: ...


class GroundedRepairAdviceProvider(Protocol):
    async def get_grounded_repair_advice(
        self, message: str, context: str
    ) -> RepairAdviceResult: ...
