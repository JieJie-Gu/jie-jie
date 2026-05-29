from __future__ import annotations

from langchain_openai import ChatOpenAI

from smart_cs.config import Settings


def configured_chat_model(settings: Settings) -> ChatOpenAI:
    return ChatOpenAI(
        model=settings.llm_model,
        api_key=settings.llm_api_key,
        base_url=settings.llm_base_url,
        temperature=0,
    )
