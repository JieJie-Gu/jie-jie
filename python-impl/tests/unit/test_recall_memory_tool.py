# 测试 recall_memory 主动检索短期上下文和筛选后的长期记忆。
from __future__ import annotations

from smart_cs.application.memory_selector import MemoryContextSelector
from smart_cs.application.memory import MemoryCandidate, MemoryDecision, MemoryWriter
from smart_cs.domain.enums import ToolCallStatus
from smart_cs.tools.agent_tool_wrappers import RuntimeToolContext, run_recall_memory
from smart_cs.tools.policy import default_tool_registry


class RecordingRepository:
    def __init__(self) -> None:
        self.calls = []

    def record_tool_call(self, **kwargs):
        self.calls.append(kwargs)


class FakeExecutor:
    def __init__(self) -> None:
        self.repository = RecordingRepository()
        self.tool_registry = default_tool_registry()


class FakeMemoryStore:
    def search(self, namespace, query: str, limit: int):
        assert namespace == ("customer", "C001", "memories")
        assert query == "之前的鞋码"
        assert limit == 20
        return [
            {
                "key": "preference:shoe_size",
                "memory_kind": "semantic",
                "memory_type": "preference",
                "title": "鞋码偏好",
                "description": "用户通常穿42码",
                "confidence": "high",
                "risk_level": "low",
                "review_status": "approved",
            },
            {
                "key": "candidate",
                "memory_kind": "semantic",
                "memory_type": "preference",
                "title": "候选",
                "description": "不应返回",
                "confidence": "high",
                "risk_level": "low",
                "review_status": "pending",
            },
            {
                "key": "episode:after_sales_event:A1",
                "memory_kind": "episodic",
                "memory_type": "after_sales_event",
                "title": "售后事件",
                "description": "用户曾提交售后",
                "confidence": "high",
                "risk_level": "low",
                "review_status": "approved",
            },
        ]


class WritableMemoryStore:
    def __init__(self) -> None:
        self.records = {}

    def put(self, namespace, key, value) -> None:
        self.records[(namespace, key)] = value

    def get(self, namespace, key):
        return self.records.get((namespace, key))

    def search(self, namespace, query: str, limit: int):
        return [
            value
            for (stored_namespace, _key), value in self.records.items()
            if stored_namespace == namespace
        ][:limit]


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


def test_recall_memory_returns_short_and_long_term_memory_and_audits() -> None:
    executor = FakeExecutor()
    ctx = RuntimeToolContext(
        conversation_id="conv-1",
        customer_id="C001",
        request_id="req-1",
        turn_fence=None,
        runtime_context={
            "session_facts": {"current_intent": "pre_sales"},
            "recent_messages": [{"role": "user", "content": "刚才那个"}],
            "conversation_summary": "用户在看鞋。",
            "pending_confirmation": None,
            "visual_evidence": {"summary": "鞋底开胶"},
        },
        memory_store=FakeMemoryStore(),
        memory_selector=MemoryContextSelector(),
    )

    result = run_recall_memory(
        executor,
        ctx,
        query="之前的鞋码",
        scope="all",
        caller_agent="PreSalesAgent",
    )

    assert result["short_term"]["session_facts"]["current_intent"] == "pre_sales"
    assert result["long_term"]["semantic_memories"][0]["memory_id"] == "preference:shoe_size"
    assert result["long_term"]["episodic_memories"]
    assert all(memory["memory_id"] != "candidate" for memory in result["long_term"]["semantic_memories"])
    assert executor.repository.calls[0]["tool_name"] == "recall_memory"
    assert executor.repository.calls[0]["status"] == ToolCallStatus.SUCCEEDED.value


def test_recall_memory_short_term_scope_does_not_search_long_term() -> None:
    executor = FakeExecutor()
    ctx = RuntimeToolContext(
        conversation_id="conv-1",
        customer_id="C001",
        request_id="req-1",
        turn_fence=None,
        runtime_context={"recent_messages": []},
        memory_store=None,
    )

    result = run_recall_memory(
        executor,
        ctx,
        query="anything",
        scope="short_term",
        caller_agent="PostSalesAgent",
    )

    assert "short_term" in result
    assert "long_term" not in result


def test_recall_memory_long_term_uses_shared_retrieval_service() -> None:
    executor = FakeExecutor()
    retrieval = RecordingMemoryRetrieval()
    ctx = RuntimeToolContext(
        conversation_id="conv-1",
        customer_id="C001",
        request_id="req-1",
        turn_fence=None,
        runtime_context={"session_facts": {"current_intent": "pre_sales"}},
        memory_retrieval=retrieval,
    )

    result = run_recall_memory(
        executor,
        ctx,
        query="shoe size",
        scope="long_term",
        caller_agent="PreSalesAgent",
    )

    assert result["long_term"]["semantic_memories"][0]["memory_id"] == "preference:shoe_size"
    assert "value" not in result["long_term"]["semantic_memories"][0]
    assert "evidence" not in result["long_term"]["semantic_memories"][0]
    assert retrieval.calls == [
        {
            "customer_id": "C001",
            "query": "shoe size",
            "intent": "pre_sales",
            "limit": 5,
            "max_chars": 1200,
        }
    ]


def test_after_sales_event_written_to_customer_memories_is_recalled_as_episodic() -> None:
    store = WritableMemoryStore()
    MemoryWriter().write(
        MemoryCandidate(
            scope="customer",
            owner_id="C001",
            memory_kind="episodic",
            memory_type="after_sales_event",
            key="episode:after_sales_event:A1:submitted",
            title="\u552e\u540e\u4e8b\u4ef6",
            description="\u7528\u6237\u63d0\u4ea4\u8ba2\u5355 O1001 \u978b\u5e95\u5f00\u80f6\u552e\u540e",
            value={"action_id": "A1", "order_id": "O1001"},
            evidence=[{"action_id": "A1"}],
            source="tool_result",
            confidence="high",
            risk_level="low",
            review_status="approved",
        ),
        MemoryDecision(action="write", reason="approved_episodic_memory"),
        store,
    )
    executor = FakeExecutor()
    ctx = RuntimeToolContext(
        conversation_id="conv-1",
        customer_id="C001",
        request_id="req-1",
        turn_fence=None,
        runtime_context={"session_facts": {"current_intent": "after_sales"}},
        memory_store=store,
        memory_selector=MemoryContextSelector(),
    )

    result = run_recall_memory(
        executor,
        ctx,
        query="\u552e\u540e O1001",
        scope="long_term",
        caller_agent="PostSalesAgent",
    )

    assert result["long_term"]["episodic_memories"][0]["memory_id"] == (
        "episode:after_sales_event:A1:submitted"
    )
