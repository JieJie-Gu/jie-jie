# 选择可安全注入 prompt 的长期记忆，并完成过滤、排序和投影。
from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, Field


class MemorySelectionInput(BaseModel):
    query: str
    intent: str | None = None
    memories: list[dict[str, Any]]
    limit: int = 5
    max_chars: int = 1200


class SelectedMemory(BaseModel):
    memory_id: str
    memory_kind: str
    memory_type: str
    title: str
    description: str
    confidence: str
    score: float


class MemorySelectionResult(BaseModel):
    memories: list[SelectedMemory] = Field(default_factory=list)


TYPE_PRIORITY = {
    "preference": 1.0,
    "constraint": 0.95,
    "profile": 0.8,
    "after_sales_event": 0.75,
    "service_event": 0.7,
    "handoff_event": 0.6,
    "complaint_event": 0.5,
    "order_event": 0.5,
}


class MemoryContextSelector:
    def select(self, input: MemorySelectionInput) -> MemorySelectionResult:
        scored = [
            (self._score(memory, input.query, input.intent), memory)
            for memory in input.memories
            if self._allowed(memory)
        ]
        ranked = sorted(scored, key=lambda item: item[0], reverse=True)
        selected: list[SelectedMemory] = []
        used_chars = 0
        for score, memory in ranked:
            item = self._project(memory, score)
            item_chars = len(item.title) + len(item.description)
            if selected and used_chars + item_chars > input.max_chars:
                break
            selected.append(item)
            used_chars += item_chars
            if len(selected) >= input.limit:
                break
        return MemorySelectionResult(memories=selected)

    def _allowed(self, memory: dict[str, Any]) -> bool:
        memory_type = str(memory.get("memory_type") or "")
        risk_level = str(memory.get("risk_level") or "")
        confidence = str(memory.get("confidence") or "")
        review_status = memory.get("review_status")
        if memory_type in {"sensitive_label", "risk_event", "badcase_candidate"}:
            return False
        if risk_level == "high" or confidence == "low":
            return False
        if review_status is None:
            review_status = self._legacy_review_status(memory)
        if review_status != "approved":
            return False
        if self._is_expired(memory.get("expires_at")):
            return False
        return True

    @staticmethod
    def _legacy_review_status(memory: dict[str, Any]) -> str:
        if (
            memory.get("confidence") == "high"
            and memory.get("risk_level") == "low"
            and memory.get("memory_type") in {"preference", "profile", "constraint"}
        ):
            return "approved"
        return "pending"

    @staticmethod
    def _is_expired(value: Any) -> bool:
        if not value:
            return False
        try:
            expires_at = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        except ValueError:
            return False
        if expires_at.tzinfo is None:
            expires_at = expires_at.replace(tzinfo=UTC)
        return expires_at <= datetime.now(UTC)

    def _score(self, memory: dict[str, Any], query: str, intent: str | None) -> float:
        relevance = self._relevance(memory, query)
        confidence = {"high": 1.0, "medium": 0.6, "low": 0.0}.get(
            str(memory.get("confidence") or "medium"),
            0.6,
        )
        recency = self._recency_score(memory)
        type_priority = TYPE_PRIORITY.get(str(memory.get("memory_type") or ""), 0.3)
        intent_match = self._intent_match(memory, intent)
        return round(
            relevance * 0.45
            + confidence * 0.2
            + recency * 0.15
            + type_priority * 0.1
            + intent_match * 0.1,
            4,
        )

    @staticmethod
    def _relevance(memory: dict[str, Any], query: str) -> float:
        haystack = " ".join(
            str(memory.get(key) or "") for key in ("title", "description", "memory_type")
        ).lower()
        terms = [term for term in str(query or "").lower().split() if term]
        if not terms:
            return 0.5
        hits = sum(1 for term in terms if term in haystack)
        return min(1.0, hits / max(1, len(terms)))

    @staticmethod
    def _recency_score(memory: dict[str, Any]) -> float:
        raw = memory.get("updated_at") or memory.get("created_at")
        if not raw:
            return 0.5
        try:
            value = datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
        except ValueError:
            return 0.5
        if value.tzinfo is None:
            value = value.replace(tzinfo=UTC)
        age_days = max(0, (datetime.now(UTC) - value).days)
        return max(0.0, 1.0 - min(age_days, 365) / 365)

    @staticmethod
    def _intent_match(memory: dict[str, Any], intent: str | None) -> float:
        if not intent:
            return 0.0
        kind = memory.get("memory_kind")
        memory_type = memory.get("memory_type")
        if intent in {"pre_sales", "product_recommendation", "product"}:
            return 1.0 if kind == "semantic" and memory_type == "preference" else 0.0
        if intent in {"after_sales", "order", "complaint"}:
            return 1.0 if kind == "episodic" else 0.0
        return 0.0

    @staticmethod
    def _project(memory: dict[str, Any], score: float) -> SelectedMemory:
        return SelectedMemory(
            memory_id=str(memory.get("memory_id") or memory.get("key") or ""),
            memory_kind=str(memory.get("memory_kind") or ""),
            memory_type=str(memory.get("memory_type") or ""),
            title=str(memory.get("title") or ""),
            description=str(memory.get("description") or ""),
            confidence=str(memory.get("confidence") or ""),
            score=score,
        )
