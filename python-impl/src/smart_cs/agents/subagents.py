from __future__ import annotations

from typing import Any

from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware

from smart_cs.infrastructure.prompts import PRE_SALES_AGENT_PROMPT, POST_SALES_AGENT_PROMPT


def create_pre_sales_agent(model: Any, tools: list[Any]):
    return create_agent(
        model,
        tools=tools,
        system_prompt=PRE_SALES_AGENT_PROMPT,
        name="pre_sales_agent",
    )


def create_post_sales_agent(model: Any, tools: list[Any]):
    return create_agent(
        model,
        tools=tools,
        system_prompt=POST_SALES_AGENT_PROMPT,
        middleware=[
            HumanInTheLoopMiddleware(
                interrupt_on={
                    "request_after_sales": True,
                    "request_handoff": True,
                },
                description_prefix="客服写动作待确认",
            )
        ],
        name="post_sales_agent",
    )
