# 测试 recall_memory 主动检索短期上下文和筛选后的长期记忆。
from __future__ import annotations

from smart_cs.application.memory_selector import MemoryContextSelector
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
