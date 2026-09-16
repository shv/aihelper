import asyncio
import json
import os

from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

from app.llm.prompts import REPAIR_ASSISTANT_INSTRUCTIONS
from main import MODEL
from scripts.demo_langchain_tool import (
    CALCULATE_FLOOR_TILES_LANGCHAIN_TOOL,
)

QUERY = (
    "Сколько нужно плиток 60 на 60 см для комнаты "
    "2.8 на 4.2 метра? Добавь запас 10 процентов."
)


async def main() -> None:
    model = ChatOpenAI(
        model=MODEL,
        api_key=SecretStr(os.environ["OPENAI_API_KEY"]),
        timeout=20.0,
        max_retries=2,
        reasoning_effort="low",
        use_responses_api=True,
        store=False,
    )

    model_with_tools = model.bind_tools(
        tools=[CALCULATE_FLOOR_TILES_LANGCHAIN_TOOL],
        tool_choice="auto",
        strict=True,
        parallel_tool_calls=False,
    )

    messages: list[BaseMessage] = [
        SystemMessage(
            content=REPAIR_ASSISTANT_INSTRUCTIONS,
        ),
        HumanMessage(
            content=QUERY,
        ),
    ]

    first_response = await model_with_tools.ainvoke(messages)

    if len(first_response.tool_calls) != 1:
        raise RuntimeError(
            f"Expected exactly one tool call, got {len(first_response.tool_calls)}"
        )

    tool_call = first_response.tool_calls[0]

    if tool_call["name"] != CALCULATE_FLOOR_TILES_LANGCHAIN_TOOL.name:
        raise RuntimeError(f"Unexpected tool: {tool_call['name']}")

    tool_result = CALCULATE_FLOOR_TILES_LANGCHAIN_TOOL.invoke(tool_call)

    if not isinstance(tool_result, ToolMessage):
        raise TypeError("Tool invocation did not return ToolMessage")

    final_response = await model_with_tools.ainvoke(
        [
            *messages,
            first_response,
            tool_result,
        ]
    )

    if final_response.tool_calls:
        raise RuntimeError("Model requested an unexpected additional tool call")

    payload = {
        "query": QUERY,
        "tool_call": tool_call,
        "tool_result": tool_result.content,
        "answer": final_response.text,
    }

    print(json.dumps(payload, ensure_ascii=False))


if __name__ == "__main__":
    asyncio.run(main())
