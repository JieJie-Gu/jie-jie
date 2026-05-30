# 使用 LLM structured output 抽取长期语义记忆和情景记忆。
from __future__ import annotations

import json
from typing import Any

from langchain_core.messages import AnyMessage, HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from smart_cs.application.memory import MemoryCandidate, MemoryExtractor
from smart_cs.infrastructure.prompts import LONG_TERM_MEMORY_EXTRACTION_PROMPT


class LongTermMemoryExtraction(BaseModel):
    facts: list[MemoryCandidate] = Field(default_factory=list)


class LongTermMemoryExtractor:
    def __init__(
        self,
        model: Any | None = None,
        fallback: MemoryExtractor | None = None,
    ) -> None:
        self.model = model
        self.fallback = fallback or MemoryExtractor()

    def extract(self, state: dict[str, Any]) -> list[dict[str, Any]]:
        if self.model is not None:
            try:
                structured = self.model.with_structured_output(LongTermMemoryExtraction)
                result = structured.invoke(self._messages(state))
                extraction = (
                    result
                    if isinstance(result, LongTermMemoryExtraction)
                    else LongTermMemoryExtraction.model_validate(result)
                )
                return [fact.model_dump() for fact in extraction.facts]
            except Exception:
                pass
        return self.fallback.extract(state)

    def _messages(self, state: dict[str, Any]) -> list[AnyMessage]:
        payload = {
            "conversation_id": state.get("conversation_id"),
            "customer_id": state.get("customer_id"),
            "current_user_message": state.get("message"),
            "recent_messages": state.get("recent_messages") or [],
            "session_facts": state.get("session_facts") or {},
            "conversation_summary": state.get("conversation_summary") or "",
            "business_result": state.get("business_result") or {},
            "pending_confirmation": state.get("pending_confirmation"),
            "visual_evidence": state.get("visual_evidence"),
            "tool_result_summary": self._tool_result_summary(state),
        }
        return [
            SystemMessage(content=LONG_TERM_MEMORY_EXTRACTION_PROMPT),
            HumanMessage(content=json.dumps(payload, ensure_ascii=False, default=str)),
        ]

    @staticmethod
    def _tool_result_summary(state: dict[str, Any]) -> dict[str, Any]:
        result = state.get("business_result") or {}
        if not isinstance(result, dict):
            return {}
        return {
            key: value
            for key, value in result.items()
            if key
            in {
                "action_id",
                "action_type",
                "status",
                "ticket_id",
                "order_id",
                "reason",
                "message",
            }
        }
