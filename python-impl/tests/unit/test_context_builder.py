# 测试运行时上下文中的摘要、active memory 和待确认动作读取。

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from smart_cs.application.context_builder import RuntimeContextBuilder, project_recent_messages
from smart_cs.infrastructure.database import Database
from smart_cs.infrastructure.repositories import SqlRepository


@dataclass
class Summary:
    summary: str


@dataclass
class Pending:
    id: str = "A1"
    customer_id: str = "C001"
    conversation_id: str = "conv-1"
    action_type: str = "after_sales"
    status: str = "pending_confirmation"
    order_id: str = "O1001"
    reason: str = "broken sole"


@dataclass
class Memory:
    id: str
    key: str
    title: str
    description: str
    confidence: str
    value_json: dict[str, Any]


class DummyRepository:
    def __init__(self) -> None:
        self.tool_calls: list[dict[str, Any]] = []

    def record_tool_call(self, **kwargs) -> None:
        self.tool_calls.append(kwargs)

    def get_conversation_summary(self, conversation_id: str, customer_id: str) -> Summary:
        assert conversation_id == "conv-1"
        assert customer_id == "C001"
        return Summary("User asked about order O1001 earlier.")

    def get_pending_action(self, conversation_id: str, customer_id: str) -> Pending:
        assert conversation_id == "conv-1"
        assert customer_id == "C001"
        return Pending()

    def list_recent_messages(self, conversation_id: str, customer_id: str, *, limit: int = 10):
        assert conversation_id == "conv-1"
        assert customer_id == "C001"
        assert limit == 10
        return [
            {
                "role": "user",
                "content": "我买的鞋开胶了，想售后",
                "content_type": "text",
                "asset_key": None,
                "visual_evidence": None,
                "created_at": "2026-05-30T10:00:00",
            },
            {
                "role": "assistant",
                "content": "请提供订单号",
                "content_type": "text",
                "asset_key": None,
                "visual_evidence": None,
                "created_at": "2026-05-30T10:01:00",
            },
            {
                "role": "user",
                "content": "O1001",
                "content_type": "text",
                "asset_key": None,
                "visual_evidence": None,
                "created_at": "2026-05-30T10:02:00",
            },
        ]


class RecordingMemoryStore:
    def __init__(self) -> None:
        self.namespaces: list[tuple[str, str, str]] = []

    def search(self, namespace, query: str, limit: int):
        self.namespaces.append(namespace)
        assert query == "need after-sales"
        assert limit == 20
        return [
            Memory(
                id="M1",
                key="shoe_size",
                title="Shoe size preference",
                description="Usually wears size 42.",
                confidence="high",
                value_json={
                    "memory_kind": "semantic",
                    "memory_type": "preference",
                    "risk_level": "low",
                    "review_status": "approved",
                },
            )
        ]


class RecordingMemoryRetrieval:
    def __init__(self) -> None:
        self.calls = []

    def search_active_memories(self, **kwargs):
        self.calls.append(kwargs)
        return [
            {
                "memory_id": "preference:shoe_size",
                "memory_kind": "semantic",
                "memory_type": "preference",
                "title": "Shoe size preference",
                "description": "Usually wears size 42.",
                "confidence": "high",
                "score": 1.0,
            }
        ]


def test_context_builder_reads_summary_active_memories_and_pending_action() -> None:
    store = RecordingMemoryStore()
    context = RuntimeContextBuilder(DummyRepository(), store).build(
        conversation_id="conv-1",
        customer_id="C001",
        message="need after-sales",
    )

    assert store.namespaces == [("customer", "C001", "memories")]
    assert context["conversation_summary"] == "User asked about order O1001 earlier."
    assert [message["content"] for message in context["recent_messages"]] == [
        "我买的鞋开胶了，想售后",
        "请提供订单号",
        "O1001",
    ]
    assert context["customer_memories"][0]["title"] == "Shoe size preference"
    assert context["pending_confirmation"]["action_id"] == "A1"


def test_context_builder_uses_injected_memory_retrieval_service() -> None:
    retrieval = RecordingMemoryRetrieval()
    repository = DummyRepository()
    context = RuntimeContextBuilder(
        repository,
        RecordingMemoryStore(),
        memory_retrieval=retrieval,
    ).build(
        conversation_id="conv-1",
        customer_id="C001",
        message="need after-sales",
    )

    assert context["customer_memories"][0]["memory_id"] == "preference:shoe_size"
    assert retrieval.calls == [
        {
            "customer_id": "C001",
            "query": "need after-sales",
            "intent": context["session_facts"].get("current_intent"),
            "limit": 5,
            "max_chars": 1200,
        }
    ]
    assert "value" not in context["customer_memories"][0]
    assert "evidence" not in context["customer_memories"][0]
    assert repository.tool_calls[-1]["arguments"]["conversation_id"] == "conv-1"


def test_context_builder_formats_compact_system_context() -> None:
    store = RecordingMemoryStore()
    builder = RuntimeContextBuilder(DummyRepository(), store)
    context = builder.build(
        conversation_id="conv-1",
        customer_id="C001",
        message="need after-sales",
    )

    text = builder.system_message(context)

    assert "Conversation summary:" in text
    assert "Recent conversation:" in text
    assert "User: 我买的鞋开胶了，想售后" in text
    assert "Assistant: 请提供订单号" in text
    assert "Active customer memories:" in text
    assert "Shoe size preference" in text
    assert "Pending confirmation:" in text
    assert "A1" in text


def test_project_recent_messages_truncates_and_projects_visual_evidence() -> None:
    rows = [
        {
            "role": "user",
            "content": "x" * 350,
            "content_type": "text",
            "asset_key": None,
            "visual_evidence": None,
            "created_at": "t1",
        },
        {
            "role": "user",
            "content": "",
            "content_type": "image",
            "asset_key": "conv-1/evidence.jpg",
            "visual_evidence": {
                "summary": "鞋底开胶图片",
                "confidence": 0.91,
                "needs_clarification": False,
            },
            "created_at": "t2",
        },
    ]

    projected = project_recent_messages(rows, max_item_chars=300)

    assert len(projected[0]["content"]) == 300
    assert projected[1]["content"] == "用户上传了图片"
    assert projected[1]["visual_evidence"]["summary"] == "鞋底开胶图片"
    assert projected[1]["visual_evidence"]["usable_for_draft"] is True


def test_sql_repository_lists_recent_messages_in_chronological_window(tmp_path) -> None:
    repository = SqlRepository(Database(f"sqlite:///{tmp_path / 'recent-messages.db'}"))
    repository.create_schema()
    repository.seed_demo_data()
    repository.claim_conversation("conv-1", "C001")
    for index in range(12):
        repository.record_message(
            "conv-1",
            "C001",
            "user" if index % 2 == 0 else "assistant",
            f"message-{index}",
        )

    rows = repository.list_recent_messages("conv-1", "C001", limit=3)

    assert [row["content"] for row in rows] == ["message-9", "message-10", "message-11"]
    assert [row["role"] for row in rows] == ["assistant", "user", "assistant"]
    assert set(rows[0]) == {
        "role",
        "content",
        "content_type",
        "asset_key",
        "visual_evidence",
        "created_at",
    }
