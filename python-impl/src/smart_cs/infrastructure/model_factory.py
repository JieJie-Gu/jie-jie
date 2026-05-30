# 根据运行时配置创建 LangChain ChatModel。

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from langchain_openai import ChatOpenAI

from smart_cs.config import Settings


@dataclass(frozen=True)
class ModelProfiles:
    agent: Any
    extraction: Any
    summary: Any
    memory: Any
    rag: Any
    vision: Any

    @classmethod
    def from_single(cls, model: Any) -> "ModelProfiles":
        return cls(
            agent=model,
            extraction=model,
            summary=model,
            memory=model,
            rag=model,
            vision=model,
        )


def configured_chat_model(settings: Settings, *, model: str | None = None) -> ChatOpenAI:
    return ChatOpenAI(
        model=model or settings.llm_model,
        api_key=settings.llm_api_key,
        base_url=settings.llm_base_url,
        temperature=0,
    )


def configured_model_profiles(settings: Settings) -> ModelProfiles:
    return ModelProfiles(
        agent=configured_chat_model(settings, model=settings.agent_model or settings.llm_model),
        extraction=configured_chat_model(
            settings,
            model=settings.extraction_model or settings.llm_model,
        ),
        summary=configured_chat_model(settings, model=settings.summary_model or settings.llm_model),
        memory=configured_chat_model(settings, model=settings.memory_model or settings.llm_model),
        rag=configured_chat_model(settings, model=settings.rag_model or settings.llm_model),
        vision=configured_chat_model(settings, model=settings.vision_model or settings.llm_model),
    )
