import asyncio
import json
import os

from langchain_openai import ChatOpenAI
from pydantic import SecretStr

from app.llm.prompts import GROUNDED_REPAIR_ASSISTANT_INSTRUCTIONS
from app.schemas import GroundedRepairAdvice

MODEL = "gpt-5.6-luna"
QUERY = "Что означает маркировка плиточного клея C2TE S1?"

CONTEXT = [
    {
        "id": "c2te-s1",
        "title": "Расшифровка аббревиатуры C2TE S1",
        "text": (
            "C — цементная основа, 2 — улучшенный класс, "
            "T — сниженное вертикальное сползание, "
            "E — увеличенное открытое время, "
            "S1 — эластичный клей."
        ),
    }
]


async def main() -> None:
    model = ChatOpenAI(
        model=MODEL,
        api_key=SecretStr(os.environ["OPENAI_API_KEY"]),
        timeout=20.0,
        max_retries=2,
        reasoning_effort="low",
        use_responses_api=True,
    )

    structured_model = model.with_structured_output(
        GroundedRepairAdvice,
        method="json_schema",
        include_raw=False,
        strict=True,
    )

    input_payload = json.dumps(
        {
            "question": QUERY,
            "context": CONTEXT,
        },
        ensure_ascii=False,
    )

    result = await structured_model.ainvoke(
        [
            (
                "system",
                GROUNDED_REPAIR_ASSISTANT_INSTRUCTIONS,
            ),
            (
                "human",
                input_payload,
            ),
        ]
    )

    if not isinstance(result, GroundedRepairAdvice):
        raise TypeError("LangChain returned unexpected structured output")

    print(result.model_dump_json())


if __name__ == "__main__":
    asyncio.run(main())
