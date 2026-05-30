# 构建摘要、长期记忆和待确认动作的紧凑上下文。

from __future__ import annotations

from typing import Any

from smart_cs.application.session_facts import SessionFactsExtractor


def project_recent_messages(
    rows: list[dict[str, Any]],
    *,
    max_item_chars: int = 300,
) -> list[dict[str, Any]]:
    projected: list[dict[str, Any]] = []
    for row in rows:
        content_type = str(row.get("content_type") or "text")
        content = str(row.get("content") or "")
        item = {
            "role": str(row.get("role") or ""),
            "content": _truncate(content, max_item_chars),
            "content_type": content_type,
            "asset_key": row.get("asset_key"),
            "created_at": row.get("created_at"),
        }
        if content_type == "image":
            item["content"] = "用户上传了图片"
            evidence = row.get("visual_evidence")
            if isinstance(evidence, dict):
                confidence = _float_or_zero(evidence.get("confidence"))
                needs_clarification = bool(evidence.get("needs_clarification"))
                item["visual_evidence"] = {
                    "summary": str(evidence.get("summary") or ""),
                    "confidence": confidence,
                    "usable_for_draft": confidence >= 0.8 and not needs_clarification,
                }
        projected.append(item)
    return projected


def _truncate(value: str, max_chars: int) -> str:
    return value[:max_chars]


def _float_or_zero(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


class RuntimeContextBuilder:
    """Build compact runtime context from approved, persisted state only."""

    def __init__(
        self,
        repository: Any,
        memory_store: Any,
        *,
        memory_limit: int = 5,
        recent_message_limit: int = 10,
        session_facts_extractor: SessionFactsExtractor | None = None,
    ) -> None:
        self.repository = repository
        self.memory_store = memory_store
        self.memory_limit = memory_limit
        self.recent_message_limit = recent_message_limit
        self.session_facts_extractor = session_facts_extractor or SessionFactsExtractor()

    def build(
        self,
        *,
        conversation_id: str,
        customer_id: str,
        message: str,
        visual_evidence: dict[str, Any] | None = None,
        asset_key: str | None = None,
    ) -> dict[str, Any]:
        summary = self._conversation_summary(conversation_id, customer_id)
        recent_messages = self._recent_messages(conversation_id, customer_id)
        session_facts = self.session_facts_extractor.extract(
            recent_messages=recent_messages,
            conversation_summary=summary,
        ).model_dump()
        memories = self._active_customer_memories(customer_id, message)
        pending = self._pending_action(conversation_id, customer_id)
        return {
            "conversation_summary": summary,
            "recent_messages": recent_messages,
            "session_facts": session_facts,
            "customer_memories": memories,
            "pending_confirmation": pending,
            "visual_evidence": visual_evidence,
            "asset_key": asset_key,
        }

    def system_message(self, context: dict[str, Any]) -> str:
        blocks: list[str] = []
        summary = context.get("conversation_summary")
        if summary:
            blocks.append("Conversation summary:\n" + str(summary))

        recent_messages = list(context.get("recent_messages") or [])
        if recent_messages:
            blocks.append(self._recent_messages_block(recent_messages))

        session_facts = context.get("session_facts")
        if isinstance(session_facts, dict) and any(
            value not in (None, "", [], {}) for value in session_facts.values()
        ):
            blocks.append(self._session_facts_block(session_facts))

        memories = list(context.get("customer_memories") or [])[: self.memory_limit]
        if memories:
            lines = ["Active customer memories:"]
            for memory in memories:
                memory_id = memory.get("memory_id") or memory.get("key") or ""
                title = memory.get("title") or ""
                description = memory.get("description") or ""
                confidence = memory.get("confidence") or ""
                lines.append(
                    f"- {memory_id}: {title}; {description}; confidence={confidence}"
                )
            blocks.append("\n".join(lines))

        pending = context.get("pending_confirmation")
        if isinstance(pending, dict) and pending:
            blocks.append(
                "Pending confirmation:\n"
                f"- action_id: {pending.get('action_id') or ''}\n"
                f"- action_type: {pending.get('action_type') or ''}\n"
                f"- status: {pending.get('status') or ''}\n"
                f"- order_id: {pending.get('order_id') or ''}\n"
                f"- reason: {pending.get('reason') or ''}"
            )

        return "\n\n".join(blocks)

    @staticmethod
    def _recent_messages_block(messages: list[dict[str, Any]]) -> str:
        lines = ["Recent conversation:"]
        role_names = {"user": "User", "human": "User", "assistant": "Assistant", "ai": "Assistant"}
        for message in messages:
            role = role_names.get(str(message.get("role") or "").lower(), str(message.get("role") or "Message"))
            lines.append(f"- {role}: {message.get('content') or ''}")
            evidence = message.get("visual_evidence")
            if isinstance(evidence, dict):
                lines.append(
                    f"  visual_evidence: {evidence.get('summary') or ''}; "
                    f"confidence={evidence.get('confidence')}; "
                    f"usable_for_draft={str(evidence.get('usable_for_draft')).lower()}"
                )
        return "\n".join(lines)

    @staticmethod
    def _session_facts_block(facts: dict[str, Any]) -> str:
        lines = ["Session facts:"]
        for key, value in facts.items():
            if value in (None, "", [], {}):
                continue
            lines.append(f"- {key}: {value}")
        return "\n".join(lines)

    def _conversation_summary(self, conversation_id: str, customer_id: str) -> str | None:
        getter = getattr(self.repository, "get_conversation_summary", None)
        if getter is None:
            return None
        summary = getter(conversation_id, customer_id)
        if summary is None:
            return None
        return str(getattr(summary, "summary", "") or "") or None

    def _active_customer_memories(self, customer_id: str, message: str) -> list[dict[str, Any]]:
        namespace = ("customer", customer_id, "memories")
        search = getattr(self.memory_store, "search", None)
        if search is None:
            return []
        records = search(namespace, query=message, limit=self.memory_limit)
        return [self._memory_record_to_dict(record) for record in records[: self.memory_limit]]

    def _recent_messages(self, conversation_id: str, customer_id: str) -> list[dict[str, Any]]:
        getter = getattr(self.repository, "list_recent_messages", None)
        if getter is None:
            return []
        rows = getter(
            conversation_id,
            customer_id,
            limit=self.recent_message_limit,
        )
        return project_recent_messages(rows)

    def _pending_action(self, conversation_id: str, customer_id: str) -> dict[str, Any] | None:
        getter = getattr(self.repository, "get_pending_action", None)
        if getter is None:
            return None
        action = getter(conversation_id, customer_id)
        if action is None:
            return None
        return {
            "action_id": action.id,
            "customer_id": action.customer_id,
            "conversation_id": action.conversation_id,
            "action_type": action.action_type,
            "status": action.status,
            "order_id": action.order_id,
            "reason": action.reason,
        }

    @staticmethod
    def _memory_record_to_dict(record: Any) -> dict[str, Any]:
        value = getattr(record, "value", None)
        if isinstance(value, dict):
            memory = dict(value)
            memory.setdefault("memory_id", getattr(record, "key", None))
            return memory

        value_json = getattr(record, "value_json", None)
        if isinstance(value_json, dict):
            memory = dict(value_json)
        else:
            memory = {}

        memory.setdefault("memory_id", getattr(record, "id", None))
        memory.setdefault("key", getattr(record, "key", None))
        memory.setdefault("title", getattr(record, "title", None))
        memory.setdefault("description", getattr(record, "description", None))
        memory.setdefault("confidence", getattr(record, "confidence", None))
        return {key: value for key, value in memory.items() if value is not None}
