from types import SimpleNamespace

from langchain_core.messages import HumanMessage

from smart_cs.application.agent_runtime import AgentRuntime
from smart_cs.application.context_projector import ContextProjector
from smart_cs.infrastructure.database import Database
from smart_cs.infrastructure.model_factory import RulesDecisionModel
from smart_cs.infrastructure.repositories import SqlRepository
from smart_cs.tools.executor import AuthorizedToolExecutor


class FakeMemoryItem:
    key = "mem-active-1"
    value = {
        "memory_type": "preference",
        "title": "Shoe size preference",
        "description": "User wears 42.",
        "confidence": "high",
        "source": "approved_memory",
    }


class RecordingStore:
    def __init__(self) -> None:
        self.calls = []

    def search(self, namespace, query: str, limit: int):
        self.calls.append((namespace, query, limit))
        return [FakeMemoryItem()]


def test_context_projector_uses_trimmed_messages_view() -> None:
    projector = ContextProjector(max_context_tokens=64)
    context = projector.build_router_context(
        {
            "message": "那我要退货",
            "messages": [
                HumanMessage(content=f"历史消息 {index}") for index in range(12)
            ],
            "conversation_slots": {"active_order_id": "O1001"},
            "customer_memories": [],
        }
    )

    assert context.current_message == "那我要退货"
    assert context.conversation_slots.active_order_id == "O1001"
    assert 0 < len(context.recent_messages) < 12


def test_runtime_context_project_reads_only_active_customer_memories(tmp_path) -> None:
    repository = SqlRepository(Database(f"sqlite:///{tmp_path / 'context.db'}"))
    repository.create_schema()
    runtime = AgentRuntime(
        executor=AuthorizedToolExecutor(repository),
        decision_model=RulesDecisionModel(),
        checkpoint_path=tmp_path / "checkpoints.db",
    )
    store = RecordingStore()
    try:
        state = {
            "conversation_id": "conv-context",
            "customer_id": "C001",
            "message": "推荐鞋码",
            "messages": [HumanMessage(content="我一般穿42码")],
        }

        result = runtime._context_project_node(state, SimpleNamespace(store=store))
    finally:
        runtime.close()

    assert store.calls == [(("customer", "C001", "memories"), "推荐鞋码", 5)]
    assert result["customer_memories"][0]["memory_id"] == "mem-active-1"
