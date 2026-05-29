from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from smart_cs.application.context_builder import RuntimeContextBuilder


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
    def get_conversation_summary(self, conversation_id: str, customer_id: str) -> Summary:
        assert conversation_id == "conv-1"
        assert customer_id == "C001"
        return Summary("User asked about order O1001 earlier.")

    def get_pending_action(self, conversation_id: str, customer_id: str) -> Pending:
        assert conversation_id == "conv-1"
        assert customer_id == "C001"
        return Pending()


class RecordingMemoryStore:
    def __init__(self) -> None:
        self.namespaces: list[tuple[str, str, str]] = []

    def search(self, namespace, query: str, limit: int):
        self.namespaces.append(namespace)
        assert query == "need after-sales"
        assert limit == 5
        return [
            Memory(
                id="M1",
                key="shoe_size",
                title="Shoe size preference",
                description="Usually wears size 42.",
                confidence="high",
                value_json={"memory_type": "preference"},
            )
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
    assert context["customer_memories"][0]["title"] == "Shoe size preference"
    assert context["pending_confirmation"]["action_id"] == "A1"


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
    assert "Active customer memories:" in text
    assert "Shoe size preference" in text
    assert "Pending confirmation:" in text
    assert "A1" in text
