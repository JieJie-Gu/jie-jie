# 抽取当前会话的结构化事实，供 supervisor 和子 Agent 稳定使用。

from __future__ import annotations

import re
from typing import Any

from pydantic import BaseModel, Field

from smart_cs.infrastructure.prompts import SESSION_FACTS_EXTRACTION_PROMPT


class SessionFacts(BaseModel):
    current_intent: str | None = None
    current_order_id: str | None = None
    current_product: str | None = None
    after_sales_reason: str | None = None
    user_constraints: list[str] = Field(default_factory=list)
    user_preferences_mentioned: list[str] = Field(default_factory=list)
    emotional_state: str | None = None
    missing_slots: list[str] = Field(default_factory=list)
    last_agent_question: str | None = None


class SessionFactsExtractor:
    def __init__(self, model: Any | None = None) -> None:
        self.model = model

    def extract(
        self,
        *,
        recent_messages: list[dict[str, Any]],
        previous_facts: dict[str, Any] | None = None,
        conversation_summary: str | None = None,
    ) -> SessionFacts:
        if self.model is not None:
            try:
                structured = self.model.with_structured_output(SessionFacts)
                result = structured.invoke(
                    {
                        "system": SESSION_FACTS_EXTRACTION_PROMPT,
                        "recent_messages": recent_messages,
                        "previous_facts": previous_facts or {},
                        "conversation_summary": conversation_summary or "",
                    }
                )
                return result if isinstance(result, SessionFacts) else SessionFacts.model_validate(result)
            except Exception:
                pass
        return self._fallback(recent_messages, previous_facts)

    def _fallback(
        self,
        recent_messages: list[dict[str, Any]],
        previous_facts: dict[str, Any] | None,
    ) -> SessionFacts:
        facts = SessionFacts.model_validate(previous_facts or {})
        text = "\n".join(str(message.get("content") or "") for message in recent_messages)
        latest_user = self._latest_by_role(recent_messages, {"user", "human"})
        last_assistant = self._latest_by_role(recent_messages, {"assistant", "ai"})

        order_match = re.search(r"\bO\d+\b", text, re.IGNORECASE)
        if order_match:
            facts.current_order_id = order_match.group(0).upper()

        after_sales_terms = ("退货", "退款", "换货", "开胶", "破损", "不合适", "售后")
        if any(term in text for term in after_sales_terms):
            facts.current_intent = "after_sales"
            facts.after_sales_reason = self._after_sales_reason(text)

        if last_assistant and ("?" in last_assistant or "？" in last_assistant or "请" in last_assistant):
            facts.last_agent_question = last_assistant

        if latest_user and any(term in latest_user for term in ("确认", "可以", "提交")):
            facts.current_intent = facts.current_intent or "confirmation"
        if latest_user and any(term in latest_user for term in ("取消", "不用了")):
            facts.current_intent = "cancellation"

        return facts

    @staticmethod
    def _latest_by_role(messages: list[dict[str, Any]], roles: set[str]) -> str:
        for message in reversed(messages):
            role = str(message.get("role") or "").lower()
            if role in roles:
                return str(message.get("content") or "")
        return ""

    @staticmethod
    def _after_sales_reason(text: str) -> str | None:
        for term in ("开胶", "破损", "不合适", "退货", "退款", "换货"):
            if term in text:
                return term
        return None
