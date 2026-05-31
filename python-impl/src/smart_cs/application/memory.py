# 提取、审核并写入长期记忆候选和会话服务事件。
from __future__ import annotations

from datetime import UTC, datetime, timedelta
import json
import logging
import re
from typing import Any, Literal, Protocol

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, RemoveMessage, SystemMessage
from pydantic import BaseModel

from smart_cs.domain.enums import ActionStatus, ToolCallStatus
from smart_cs.infrastructure.prompts import CONVERSATION_ROLLING_SUMMARY_PROMPT
from smart_cs.application.memory_retrieval import MemoryVectorIndex, is_indexable_memory


LOGGER = logging.getLogger(__name__)


class MemoryStoreProtocol(Protocol):
    def put(self, namespace: tuple[str, str, str], key: str, value: dict[str, Any]) -> None: ...

    def get(self, namespace: tuple[str, str, str], key: str) -> Any | None: ...

    def get_by_id(self, memory_id: str) -> Any | None: ...

    def search(self, namespace: tuple[str, str, str], query: str, limit: int) -> list[Any]: ...


class MemoryDecision(BaseModel):
    action: Literal["write", "candidate", "human_review", "discard"]
    reason: str


class MemoryCandidate(BaseModel):
    scope: Literal["customer", "conversation", "tenant"]
    owner_id: str
    memory_kind: Literal["semantic", "episodic"]
    memory_type: Literal[
        "preference",
        "profile",
        "constraint",
        "service_event",
        "after_sales_event",
        "handoff_event",
        "complaint_event",
        "order_event",
        "risk_event",
        "sensitive_label",
        "badcase_candidate",
    ]
    key: str
    title: str
    description: str
    value: dict[str, Any]
    evidence: list[dict[str, Any]]
    source: Literal["llm_extraction", "tool_result", "user_message", "system", "human_review"]
    confidence: Literal["low", "medium", "high"]
    risk_level: Literal["low", "medium", "high"]
    review_status: Literal["pending", "approved", "rejected"] = "pending"
    expires_at: str | None = None
    created_at: str | None = None
    updated_at: str | None = None


def normalize_memory_candidate(
    raw: dict[str, Any],
    *,
    customer_id: str | None = None,
    conversation_id: str | None = None,
) -> dict[str, Any]:
    candidate = dict(raw)
    memory_type = str(candidate.get("memory_type") or "preference")
    if not candidate.get("memory_kind"):
        candidate["memory_kind"] = (
            "episodic"
            if memory_type
            in {
                "service_event",
                "after_sales_event",
                "handoff_event",
                "complaint_event",
                "order_event",
            }
            else "semantic"
        )
    source = str(candidate.get("source") or "system")
    if source in {"pending_action", "confirmed_action"}:
        candidate["source"] = "tool_result"
    if candidate.get("memory_kind") == "episodic" and customer_id:
        candidate["scope"] = "customer"
        candidate["owner_id"] = customer_id
        candidate.setdefault("value", {})
        if isinstance(candidate["value"], dict):
            candidate["value"].setdefault("conversation_id", conversation_id)
    candidate.setdefault("review_status", "pending")
    candidate.setdefault("evidence", [])
    candidate.setdefault("value", {})
    return candidate


class MemoryPolicy:
    def decide(self, candidate: dict[str, Any]) -> MemoryDecision:
        kind = candidate.get("memory_kind")
        memory_type = candidate.get("memory_type")
        confidence = candidate.get("confidence")
        risk_level = candidate.get("risk_level")

        if memory_type in {"sensitive_label", "risk_event", "badcase_candidate"}:
            return MemoryDecision(
                action="human_review",
                reason="sensitive_or_risk_memory_requires_review",
            )
        if risk_level == "high":
            return MemoryDecision(action="human_review", reason="high_risk_memory_requires_review")
        if confidence == "low":
            return MemoryDecision(action="candidate", reason="low_confidence_memory_candidate")

        if kind == "semantic" and memory_type in {"preference", "profile", "constraint"}:
            if confidence == "high" and risk_level == "low":
                return MemoryDecision(action="write", reason="approved_semantic_memory")
            return MemoryDecision(action="candidate", reason="semantic_memory_needs_more_evidence")

        if kind == "episodic":
            if memory_type in {
                "service_event",
                "after_sales_event",
                "handoff_event",
                "order_event",
            } and risk_level in {"low", "medium"}:
                return MemoryDecision(action="write", reason="approved_episodic_memory")
            if memory_type == "complaint_event":
                return MemoryDecision(action="candidate", reason="complaint_event_needs_review")

        return MemoryDecision(action="discard", reason="unsupported_memory")


PREFERENCE_PATTERNS = [
    ("shoe_size", re.compile(r"(?:我(?:一般|通常)?穿|鞋码(?:是)?)(\d{2})码?")),
    ("color_preference", re.compile(r"(?:我(?:喜欢|偏好)|偏好)(黑色|白色|灰色|蓝色)")),
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
        memory_type = {
            "after_sales": "after_sales_event",
            "handoff": "handoff_event",
        }.get(str(action_type), "service_event")
        return [
            MemoryCandidate(
                scope="customer",
                owner_id=state["customer_id"],
                memory_kind="episodic",
                memory_type=memory_type,
                key=f"episode:{memory_type}:{action_id}:{status}",
                title=f"{action_type} {status}",
                description=f"Conversation action {action_id} reached {status}.",
                value={
                    **{
                        key: value
                        for key, value in result.items()
                        if key in {"action_id", "action_type", "status", "ticket_id", "order_id"}
                    },
                    "conversation_id": state["conversation_id"],
                },
                evidence=[
                    {
                        "conversation_id": state["conversation_id"],
                        "customer_id": state["customer_id"],
                        "business_result": result,
                    }
                ],
                source="tool_result",
                confidence="high",
                risk_level="low",
                review_status="approved",
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
                    memory_kind="semantic",
                    memory_type="preference",
                    key=f"preference:{key}",
                    title=self._preference_title(key, raw_value),
                    description=f"User explicitly stated preference {key}={raw_value}.",
                    value={key: raw_value},
                    evidence=[{"text": message, "conversation_id": state["conversation_id"]}],
                    source="user_message",
                    confidence="high",
                    risk_level="low",
                    review_status="approved",
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
    def __init__(self, repository: Any, memory_index: MemoryVectorIndex | None = None) -> None:
        self.repository = repository
        self.memory_index = memory_index

    def put(self, namespace: tuple[str, str, str], key: str, value: dict[str, Any]) -> Any:
        scope, owner_id, _bucket = namespace
        record = self.repository.put_memory(
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
        self._sync_index(record, namespace, value)
        return record

    def get(self, namespace: tuple[str, str, str], key: str) -> Any | None:
        getter = getattr(self.repository, "get_memory", None)
        if getter is None:
            return None
        return getter(namespace, key)

    def get_by_id(self, memory_id: str) -> Any | None:
        getter = getattr(self.repository, "get_memory_by_id", None)
        if getter is None:
            return None
        return getter(memory_id)

    def search(self, namespace: tuple[str, str, str], query: str, limit: int) -> list[Any]:
        return self.repository.search_memories(namespace, query=query, limit=limit)

    def _sync_index(
        self,
        record: Any,
        namespace: tuple[str, str, str],
        value: dict[str, Any],
    ) -> None:
        if self.memory_index is None:
            return
        try:
            payload = {
                **value,
                "memory_id": getattr(record, "id", None),
                "id": getattr(record, "id", None),
                "namespace": "/".join(namespace),
                "key": key_from_record(record, value),
                "scope": namespace[0],
                "owner_id": namespace[1],
            }
            if is_indexable_memory(payload):
                self.memory_index.upsert(payload)
            else:
                self.memory_index.delete(payload)
        except Exception:
            LOGGER.warning("Unable to sync memory vector index", exc_info=True)


def key_from_record(record: Any, value: dict[str, Any]) -> str:
    return str(getattr(record, "key", None) or value.get("key") or "")


class MemoryWriter:
    DEFAULT_MEMORY_TTL_DAYS = {
        "preference": 365,
        "profile": 365,
        "constraint": 365,
        "service_event": 180,
        "after_sales_event": 365,
        "handoff_event": 180,
        "complaint_event": 365,
        "order_event": 180,
    }

    def write(
        self,
        candidate: MemoryCandidate,
        decision: MemoryDecision,
        store: MemoryStoreProtocol,
    ) -> list[dict[str, Any]]:
        if decision.action == "discard":
            return [
                {
                    "operation": "discard",
                    "key": candidate.key,
                    "reason": decision.reason,
                    "before_json": None,
                    "after_json": None,
                }
            ]
        namespace = self._namespace_for(candidate, decision)
        if decision.action == "write":
            if candidate.memory_kind == "semantic":
                return self._upsert_semantic(candidate, decision, namespace, store)
            if candidate.memory_kind == "episodic":
                return self._append_episodic(candidate, decision, namespace, store)
        return self._write_candidate(candidate, decision, namespace, store)

    @staticmethod
    def _namespace_for(
        candidate: MemoryCandidate,
        decision: MemoryDecision,
    ) -> tuple[str, str, str]:
        if candidate.memory_type == "badcase_candidate":
            return ("tenant", candidate.owner_id or "default", "badcase_candidates")
        if decision.action == "write":
            return (candidate.scope, candidate.owner_id, "memories")
        return (candidate.scope, candidate.owner_id, "memory_candidates")

    def _upsert_semantic(
        self,
        candidate: MemoryCandidate,
        decision: MemoryDecision,
        namespace: tuple[str, str, str],
        store: MemoryStoreProtocol,
    ) -> list[dict[str, Any]]:
        value = self._payload(candidate, decision)
        existing = self._stored_value(store.get(namespace, candidate.key))
        if existing:
            old_value = existing.get("value") if isinstance(existing.get("value"), dict) else {}
            if old_value == candidate.value or old_value.get("current") == candidate.value:
                value["evidence"] = self._merge_evidence(existing.get("evidence", []), value["evidence"])
                value["confidence"] = self._higher_confidence(
                    str(existing.get("confidence") or "medium"),
                    value["confidence"],
                )
            else:
                candidate_namespace = (candidate.scope, candidate.owner_id, "memory_candidates")
                candidate_value = self._payload(candidate, decision)
                candidate_value["value"] = {
                    "proposed_value": candidate.value,
                    "previous_value": old_value,
                    "conflict": True,
                    "conflict_with": candidate.key,
                }
                candidate_value["confidence"] = "medium"
                candidate_value["review_status"] = "pending"
                candidate_value["conflict"] = True
                candidate_value["evidence"] = self._merge_evidence(
                    existing.get("evidence", []),
                    candidate_value["evidence"],
                )
                store.put(candidate_namespace, candidate.key, candidate_value)
                return [
                    {
                        "operation": "semantic_conflict",
                        "key": candidate.key,
                        "namespace": candidate_namespace,
                        "reason": decision.reason,
                        "before_json": existing,
                        "after_json": candidate_value,
                    }
                ]
        store.put(namespace, candidate.key, value)
        return [
            {
                "operation": "semantic_upsert",
                "key": candidate.key,
                "namespace": namespace,
                "reason": decision.reason,
                "before_json": existing or None,
                "after_json": value,
            }
        ]

    def _append_episodic(
        self,
        candidate: MemoryCandidate,
        decision: MemoryDecision,
        namespace: tuple[str, str, str],
        store: MemoryStoreProtocol,
    ) -> list[dict[str, Any]]:
        value = self._payload(candidate, decision)
        existing = self._stored_value(store.get(namespace, candidate.key))
        if existing:
            value["evidence"] = self._merge_evidence(existing.get("evidence", []), value["evidence"])
        store.put(namespace, candidate.key, value)
        return [
            {
                "operation": "episodic_append",
                "key": candidate.key,
                "namespace": namespace,
                "reason": decision.reason,
                "before_json": existing or None,
                "after_json": value,
            }
        ]

    def _write_candidate(
        self,
        candidate: MemoryCandidate,
        decision: MemoryDecision,
        namespace: tuple[str, str, str],
        store: MemoryStoreProtocol,
    ) -> list[dict[str, Any]]:
        value = self._payload(candidate, decision)
        existing = self._stored_value(store.get(namespace, candidate.key))
        store.put(namespace, candidate.key, value)
        return [
            {
                "operation": "candidate_write",
                "key": candidate.key,
                "namespace": namespace,
                "reason": decision.reason,
                "before_json": existing or None,
                "after_json": value,
            }
        ]

    def _payload(self, candidate: MemoryCandidate, decision: MemoryDecision) -> dict[str, Any]:
        value = candidate.model_dump()
        now = datetime.now(UTC).isoformat()
        value["created_at"] = value.get("created_at") or now
        value["updated_at"] = now
        value["expires_at"] = value.get("expires_at") or self._default_expires_at(candidate.memory_type)
        value["memory_decision"] = decision.model_dump()
        return value

    def _default_expires_at(self, memory_type: str) -> str | None:
        days = self.DEFAULT_MEMORY_TTL_DAYS.get(memory_type)
        if days is None:
            return None
        return (datetime.now(UTC) + timedelta(days=days)).isoformat()

    @staticmethod
    def _stored_value(record: Any | None) -> dict[str, Any]:
        if record is None:
            return {}
        if isinstance(record, dict):
            return dict(record)
        value = getattr(record, "value", None)
        if isinstance(value, dict):
            return dict(value)
        value_json = getattr(record, "value_json", None)
        return dict(value_json) if isinstance(value_json, dict) else {}

    @staticmethod
    def _merge_evidence(existing: Any, incoming: list[dict[str, Any]]) -> list[dict[str, Any]]:
        merged: list[dict[str, Any]] = []
        seen: set[str] = set()
        for item in list(existing or []) + incoming:
            if not isinstance(item, dict):
                continue
            marker = json.dumps(item, sort_keys=True, ensure_ascii=False, default=str)
            if marker in seen:
                continue
            seen.add(marker)
            merged.append(item)
        return merged

    @staticmethod
    def _higher_confidence(left: str, right: str) -> str:
        order = {"low": 0, "medium": 1, "high": 2}
        return left if order.get(left, 0) >= order.get(right, 0) else right


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

    @property
    def can_remove_messages(self) -> bool:
        return self.summarizer is not None

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
            try:
                response = self.summarizer.invoke(
                    self._messages(
                        existing_summary=current,
                        new_messages=removed_text,
                        business_result=business_result,
                    )
                )
            except Exception:
                return current
            text = getattr(response, "content", str(response))
        elif business_result:
            text = current or str(business_result)
        else:
            text = current
        return text[-self.max_summary_chars:] if text else current

    @staticmethod
    def _messages(
        *,
        existing_summary: str,
        new_messages: str,
        business_result: dict[str, Any],
    ) -> list[AnyMessage]:
        payload = {
            "existing_summary": existing_summary,
            "new_messages": new_messages,
            "business_result": business_result,
        }
        return [
            SystemMessage(content=CONVERSATION_ROLLING_SUMMARY_PROMPT),
            HumanMessage(content=json.dumps(payload, ensure_ascii=False, default=str)),
        ]


class MemoryWriteback:
    def __init__(
        self,
        *,
        repository: Any,
        summarizer: ConversationSummarizer | None = None,
        extractor: Any | None = None,
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
        summary = self.summarizer.summarize(state, removable)
        remove_messages = (
            [RemoveMessage(id=str(message.id)) for message in removable]
            if self.summarizer.can_remove_messages
            and summary != str(state.get("conversation_summary") or "").strip()
            else []
        )
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

        raw_candidates = self.extractor.extract(state)
        self._record_tool_audit(
            "long_term_memory_extract",
            state,
            result={"count": len(raw_candidates), "candidates": raw_candidates},
        )

        for raw_candidate in raw_candidates:
            candidate = MemoryCandidate.model_validate(
                normalize_memory_candidate(
                    raw_candidate,
                    customer_id=customer_id,
                    conversation_id=conversation_id,
                )
            )
            decision = self.policy.decide(candidate.model_dump())
            self._record_tool_audit(
                "memory_policy_decide",
                state,
                result={
                    "key": candidate.key,
                    "memory_kind": candidate.memory_kind,
                    "memory_type": candidate.memory_type,
                    "decision": decision.model_dump(),
                },
            )
            write_results = self.writer.write(candidate, decision, store)
            for write_result in write_results:
                self._record_tool_audit(
                    "memory_conflict"
                    if write_result.get("operation") == "semantic_conflict"
                    else "memory_write",
                    state,
                    result={
                        "key": candidate.key,
                        "memory_kind": candidate.memory_kind,
                        "memory_type": candidate.memory_type,
                        "decision": decision.model_dump(),
                        "operation": write_result.get("operation"),
                        "reason": write_result.get("reason"),
                        "before_json": write_result.get("before_json"),
                        "after_json": write_result.get("after_json"),
                    },
                )

        return {"conversation_summary": summary, "messages": remove_messages}

    def _record_tool_audit(
        self,
        tool_name: str,
        state: dict[str, Any],
        *,
        result: dict[str, Any],
    ) -> None:
        recorder = getattr(self.repository, "record_tool_call", None)
        if recorder is None:
            return
        try:
            recorder(
                tool_name=tool_name,
                arguments={
                    "conversation_id": state.get("conversation_id"),
                    "customer_id": state.get("customer_id"),
                    "request_id": state.get("request_id"),
                },
                customer_id=state.get("customer_id"),
                status=ToolCallStatus.SUCCEEDED.value,
                result=result,
            )
        except Exception:
            return
