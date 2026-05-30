# 测试 AgentRuntime 的 HITL resume、ContextVar 和记忆 checkpoint helper。

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from contextvars import ContextVar
from dataclasses import dataclass
from threading import Barrier
from typing import Any

from langgraph.types import Command

from smart_cs.agents.state import RuntimeState
from smart_cs.application.agent_runtime import AgentRuntime
from smart_cs.infrastructure.model_factory import ModelProfiles
from smart_cs.tools.agent_tool_wrappers import RuntimeToolContext


class RecordingGraph:
    def __init__(self, *, fail_first: bool = False) -> None:
        self.fail_first = fail_first
        self.invocations: list[tuple[Any, dict[str, dict[str, str]]]] = []

    def invoke(self, command: Command, *, config: dict[str, dict[str, str]]) -> dict[str, Any]:
        self.invocations.append((command.resume, config))
        if self.fail_first and len(self.invocations) == 1:
            raise KeyError("decisions")
        return {"messages": []}


class RecordingUpdateGraph:
    def __init__(self) -> None:
        self.updates: list[tuple[dict[str, dict[str, str]], dict[str, Any]]] = []

    def update_state(
        self,
        config: dict[str, dict[str, str]],
        updates: dict[str, Any],
    ) -> None:
        self.updates.append((config, updates))


class FakeMemoryWriteback:
    def __init__(self) -> None:
        self.state = None

    def update(self, _state: dict[str, Any], *, store: Any) -> dict[str, Any]:
        self.state = _state
        return {"conversation_summary": "summary", "messages": []}


class FakeExecutor:
    repository = object()


@dataclass
class FakeInterrupt:
    id: str
    value: dict[str, Any]


def runtime_shell() -> AgentRuntime:
    runtime = AgentRuntime.__new__(AgentRuntime)
    runtime._current_tool_context_var = ContextVar("test_tool_context", default=None)
    return runtime


def test_resume_uses_keyed_interrupt_payload_first() -> None:
    runtime = runtime_shell()
    runtime.graph = RecordingGraph()
    ctx = RuntimeToolContext("conv-1", "C001", "req-1", None)

    runtime._resume_with_decision(
        "conv-1",
        ctx,
        {"type": "approve"},
        interrupt_id="interrupt-1",
    )

    assert runtime.graph.invocations[0][0] == {
        "interrupt-1": {"decisions": [{"type": "approve"}]}
    }


def test_resume_falls_back_to_unkeyed_payload_for_local_middleware_shape() -> None:
    runtime = runtime_shell()
    runtime.graph = RecordingGraph(fail_first=True)
    ctx = RuntimeToolContext("conv-1", "C001", "req-1", None)

    runtime._resume_with_decision(
        "conv-1",
        ctx,
        {"type": "reject", "message": "cancel"},
        interrupt_id="interrupt-1",
    )

    assert runtime.graph.invocations[0][0] == {
        "interrupt-1": {"decisions": [{"type": "reject", "message": "cancel"}]}
    }
    assert runtime.graph.invocations[1][0] == {
        "decisions": [{"type": "reject", "message": "cancel"}]
    }


def test_interrupt_result_preserves_interrupt_id() -> None:
    runtime = runtime_shell()
    runtime._write_memory = lambda **_kwargs: None
    runtime._draft_action_for_interrupt = lambda _request, _ctx: {
        "action_id": "A1",
        "action_type": "after_sales",
        "status": "pending_confirmation",
    }
    ctx = RuntimeToolContext("conv-1", "C001", "req-1", None)

    result = runtime._result_from_interrupt(
        {
            "__interrupt__": [
                FakeInterrupt(
                    id="interrupt-1",
                    value={
                        "action_requests": [
                            {
                                "name": "request_after_sales",
                                "args": {"order_id": "O1001", "reason": "broken"},
                            }
                        ]
                    },
                )
            ],
            "messages": [],
        },
        ctx,
        message="broken",
        fallback_messages=[],
    )

    assert result["status"] == "pending_confirmation"
    assert result["interrupt_id"] == "interrupt-1"


def test_tool_context_uses_contextvars_without_cross_thread_leakage() -> None:
    runtime = runtime_shell()
    barrier = Barrier(2)

    def read_context(conversation_id: str, customer_id: str) -> tuple[str, str]:
        ctx = RuntimeToolContext(conversation_id, customer_id, f"{conversation_id}:req", None)
        with runtime._tool_context(ctx):
            barrier.wait(timeout=5)
            active = runtime._current_tool_context()
            return active.conversation_id, active.customer_id

    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(read_context, "conv-1", "C001")
        second = pool.submit(read_context, "conv-2", "C002")

    assert first.result(timeout=5) == ("conv-1", "C001")
    assert second.result(timeout=5) == ("conv-2", "C002")


def test_write_memory_applies_summary_to_graph_checkpoint() -> None:
    runtime = runtime_shell()
    runtime.graph = RecordingUpdateGraph()
    runtime.memory_writeback = FakeMemoryWriteback()
    runtime.store = object()

    runtime._write_memory(
        conversation_id="conv-1",
        customer_id="C001",
        request_id="req-1",
        message="hello",
        messages=[],
        business_result=None,
    )

    assert runtime.graph.updates == [
        (
            {"configurable": {"thread_id": "conv-1"}},
            {"conversation_summary": "summary"},
        )
    ]


def test_write_memory_passes_runtime_context_to_long_term_extractor() -> None:
    runtime = runtime_shell()
    runtime.graph = RecordingUpdateGraph()
    runtime.memory_writeback = FakeMemoryWriteback()
    runtime.store = object()

    runtime._write_memory(
        conversation_id="conv-1",
        customer_id="C001",
        request_id="req-1",
        message="我一般穿42码",
        messages=[],
        business_result=None,
        runtime_context={
            "recent_messages": [{"role": "user", "content": "我一般穿42码"}],
            "session_facts": {"current_intent": "pre_sales"},
            "conversation_summary": "summary",
            "customer_memories": [],
            "pending_confirmation": None,
            "visual_evidence": {"summary": "evidence"},
            "asset_key": "asset-1",
        },
    )

    assert runtime.memory_writeback.state["recent_messages"][0]["content"] == "我一般穿42码"
    assert runtime.memory_writeback.state["session_facts"]["current_intent"] == "pre_sales"
    assert runtime.memory_writeback.state["visual_evidence"]["summary"] == "evidence"


def test_runtime_state_does_not_define_duplicate_message_field() -> None:
    assert "message" not in RuntimeState.__annotations__
    assert "messages" in RuntimeState.__annotations__
    assert "recent_messages" in RuntimeState.__annotations__
    assert "session_facts" in RuntimeState.__annotations__


def test_agent_runtime_wires_model_profiles_into_context_builder(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(AgentRuntime, "_build_supervisor", lambda self: object())
    profiles = ModelProfiles(
        agent=object(),
        extraction=object(),
        summary=object(),
        memory=object(),
        rag=object(),
        vision=object(),
    )

    runtime = AgentRuntime(
        executor=FakeExecutor(),
        checkpoint_path=tmp_path / "checkpoint.sqlite",
        model_profiles=profiles,
    )
    try:
        assert runtime.chat_model is profiles.agent
        assert runtime.model_profiles is profiles
        assert runtime.context_builder.session_facts_extractor.model is profiles.extraction
    finally:
        runtime.close()
