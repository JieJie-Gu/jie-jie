from __future__ import annotations

import re
from typing import Any, Literal, Protocol

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, RemoveMessage
from pydantic import BaseModel

from smart_cs.domain.enums import ActionStatus


class MemoryStoreProtocol(Protocol):
    def put(self, namespace: tuple[str, str, str], key: str, value: dict[str, Any]) -> None: ...

    def search(self, namespace: tuple[str, str, str], query: str, limit: int) -> list[Any]: ...


class MemoryDecision(BaseModel):
    action: Literal["write", "candidate", "human_review", "discard"]
    reason: str


class MemoryCandidate(BaseModel):
    scope: Literal["customer", "conversation", "tenant"]
    owner_id: str
    memory_type: Literal[
        "preference", "service_event", "risk_event", "sensitive_label", "badcase_candidate"
    ]
    key: str
    title: str
    description: str
    value: dict[str, Any]
    evidence: list[dict[str, Any]]
    source: str
    confidence: Literal["low", "medium", "high"]
    risk_level: Literal["low", "medium", "high"]
    review_status: Literal["pending", "approved", "rejected"] = "pending"


class MemoryPolicy:
    def decide(self, candidate: dict[str, Any]) -> MemoryDecision:
        memory_type = candidate.get("memory_type")
        risk_level = candidate.get("risk_level")
        if memory_type == "service_event" and risk_level == "low":
            return MemoryDecision(action="write", reason="low_risk_service_event")
        if memory_type == "preference":
            return MemoryDecision(action="candidate", reason="user_preference_candidate")
        if memory_type in {"sensitive_label", "risk_label", "risk_event"}:
            return MemoryDecision(action="human_review", reason="sensitive_memory")
        if memory_type == "badcase_candidate":
            return MemoryDecision(action="human_review", reason="badcase_requires_review")
        return MemoryDecision(action="discard", reason="unsupported_memory")


PREFERENCE_PATTERNS = [
    ("shoe_size", re.compile(r"我(?:一般|通常)?穿(\d{2})码")),
    ("color_preference", re.compile(r"我(?:喜欢|偏好)(黑色|白色|灰色|蓝色)")),
    ("contact_preference", re.compile(r"(?:以后|之后).*(?:别|不要).*打电话")),
]


class MemoryExtractor:
    def extract(self, state: dict[str, Any]) -> list[dict[str, Any]]:
        candidates: list[MemoryCandidate] = []
        candidates.extend(self._extract_service_events(state))
        candidates.extend(self._extract_explicit_preferences(state))
        return [candidate.model_dump() for candidate in candidates]

    def _extract_service_events(self, state: dict[str, Any]) -> list[MemoryCandidate]:
        result = state.get("business_result") or {}
        action_id = result.get("action_id")
        action_type = result.get("action_type")
        status = result.get("status")
        if action_id is None or action_type is None or status not in {
            ActionStatus.PENDING_CONFIRMATION.value,
            ActionStatus.SUBMITTED.value,
            ActionStatus.CANCELLED.value,
        }:
            return []
        source = "pending_action" if status == ActionStatus.PENDING_CONFIRMATION.value else "confirmed_action"
        return [
            MemoryCandidate(
                scope="conversation",
                owner_id=state["conversation_id"],
                memory_type="service_event",
                key=f"{action_type}:{action_id}:{status}",
                title=f"{action_type} {status}",
                description=f"Conversation action {action_id} reached {status}.",
                value={
                    key: value
                    for key, value in result.items()
                    if key in {"action_id", "action_type", "status", "ticket_id", "order_id"}
                },
                evidence=[
                    {
                        "conversation_id": state["conversation_id"],
                        "customer_id": state["customer_id"],
                        "business_result": result,
                    }
                ],
                source=source,
                confidence="high",
                risk_level="low",
            )
        ]

    def _extract_explicit_preferences(self, state: dict[str, Any]) -> list[MemoryCandidate]:
        message = str(state.get("message") or "")
        if not message:
            return []
        candidates: list[MemoryCandidate] = []
        for key, pattern in PREFERENCE_PATTERNS:
            match = pattern.search(message)
            if match is None:
                continue
            raw_value = match.group(1) if match.groups() else "no_phone_call"
            candidates.append(
                MemoryCandidate(
                    scope="customer",
                    owner_id=state["customer_id"],
                    memory_type="preference",
                    key=key,
                    title=self._preference_title(key, raw_value),
                    description=f"User explicitly stated preference {key}={raw_value}.",
                    value={key: raw_value},
                    evidence=[{"text": message, "conversation_id": state["conversation_id"]}],
                    source="user_message",
                    confidence="high",
                    risk_level="low",
                )
            )
        return candidates

    @staticmethod
    def _preference_title(key: str, raw_value: str) -> str:
        if key == "shoe_size":
            return f"Shoe size preference: {raw_value}"
        if key == "color_preference":
            return f"Color preference: {raw_value}"
        if key == "contact_preference":
            return "Contact preference: no phone call"
        return f"Preference: {key}"


class SqlMemoryStoreAdapter:
    def __init__(self, repository: Any) -> None:
        self.repository = repository

    def put(self, namespace: tuple[str, str, str], key: str, value: dict[str, Any]) -> None:
        scope, owner_id, _bucket = namespace
        self.repository.put_memory(
            namespace,
            key,
            value,
            scope=scope,
            owner_id=owner_id,
            memory_type=str(value.get("memory_type", "service_event")),
            source=str(value.get("source", "system")),
            confidence=str(value.get("confidence", "medium")),
            risk_level=str(value.get("risk_level", "low")),
            created_by="system",
        )

    def search(self, namespace: tuple[str, str, str], query: str, limit: int) -> list[Any]:
        return self.repository.search_memories(namespace, query=query, limit=limit)


class MemoryWriter:
    def write(
        self,
        candidate: MemoryCandidate,
        decision: MemoryDecision,
        store: MemoryStoreProtocol,
    ) -> None:
        if decision.action == "discard":
            return
        namespace = self._namespace_for(candidate, decision)
        value = candidate.model_dump()
        value["memory_decision"] = decision.model_dump()
        store.put(namespace, candidate.key, value)

    @staticmethod
    def _namespace_for(
        candidate: MemoryCandidate,
        decision: MemoryDecision,
    ) -> tuple[str, str, str]:
        if candidate.memory_type == "service_event" and decision.action == "write":
            return ("conversation", candidate.owner_id, "events")
        if candidate.memory_type == "badcase_candidate":
            return ("tenant", candidate.owner_id or "default", "badcase_candidates")
        if decision.action == "write":
            return (candidate.scope, candidate.owner_id, "memories")
        return (candidate.scope, candidate.owner_id, "memory_candidates")


class ConversationSummarizer:
    def __init__(
        self,
        *,
        summary_keep_last: int = 6,
        max_summary_chars: int = 2000,
        summarizer: Any | None = None,
    ) -> None:
        self.summary_keep_last = summary_keep_last
        self.max_summary_chars = max_summary_chars
        self.summarizer = summarizer

    def removable_messages(self, messages: list[AnyMessage]) -> list[AnyMessage]:
        if len(messages) <= self.summary_keep_last:
            return []
        return [
            message
            for message in messages[: -self.summary_keep_last]
            if isinstance(message, (HumanMessage, AIMessage)) and getattr(message, "id", None) is not None
        ]

    def summarize(self, state: dict[str, Any], removable: list[AnyMessage]) -> str:
        current = str(state.get("conversation_summary") or "").strip()
        removed_text = "\n".join(str(message.content) for message in removable)
        business_result = state.get("business_result") or {}
        if self.summarizer is not None and removed_text:
            response = self.summarizer.invoke(
                {
                    "existing_summary": current,
                    "new_messages": removed_text,
                    "business_result": business_result,
                }
            )
            text = getattr(response, "content", str(response))
        else:
            parts = [
                part for part in [current, removed_text, str(business_result) if business_result else ""] if part
            ]
            text = "\n".join(parts)
        return text[-self.max_summary_chars:] if text else current


class MemoryWriteback:
    def __init__(
        self,
        *,
        repository: Any,
        summarizer: ConversationSummarizer | None = None,
        extractor: MemoryExtractor | None = None,
        policy: MemoryPolicy | None = None,
        writer: MemoryWriter | None = None,
    ) -> None:
        self.repository = repository
        self.summarizer = summarizer or ConversationSummarizer()
        self.extractor = extractor or MemoryExtractor()
        self.policy = policy or MemoryPolicy()
        self.writer = writer or MemoryWriter()

    def update(self, state: dict[str, Any], *, store: MemoryStoreProtocol) -> dict[str, Any]:
        conversation_id = state["conversation_id"]
        customer_id = state["customer_id"]
        messages = list(state.get("messages") or [])
        removable = self.summarizer.removable_messages(messages)
        remove_messages = [RemoveMessage(id=str(message.id)) for message in removable]
        summary = self.summarizer.summarize(state, removable)
        route = state.get("route") or {}
        if summary:
            self.repository.upsert_conversation_summary(
                conversation_id,
                customer_id,
                summary,
                open_items={},
                last_intent=route.get("intent"),
                last_entities=route.get("entities") or {},
            )

        for raw_candidate in self.extractor.extract(state):
            candidate = MemoryCandidate.model_validate(raw_candidate)
            decision = self.policy.decide(candidate.model_dump())
            self.writer.write(candidate, decision, store)

        return {"conversation_summary": summary, "messages": remove_messages}
