# 测试 supervisor 高级子 Agent 工具的上下文传递。

from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage

from smart_cs.tools.subagent_tools import make_post_sales_tool, make_pre_sales_tool


class FakeRuntime:
    state = {"messages": [HumanMessage(content="订单 O1001 鞋底开胶，帮我申请售后")]}
    config = {"configurable": {"thread_id": "conv-1"}}


class FakeRuntimeWithVisual:
    state = {
        "messages": [HumanMessage(content="订单 O1001 鞋底开胶，帮我申请售后")],
        "visual_evidence": {
            "summary": "图片不清晰，无法确认鞋底问题",
            "confidence": 0.42,
            "needs_clarification": True,
        },
        "asset_key": "conv-1/evidence.jpg",
    }
    config = {"configurable": {"thread_id": "conv-1"}}


class FakeRuntimeWithCompactContext:
    state = {
        "messages": [HumanMessage(content="订单 O1001 鞋底开胶，帮我申请售后")],
        "conversation_summary": "User previously asked about order O1001.",
        "recent_messages": [
            {"role": "user", "content": "我买的鞋开胶了，想售后"},
            {"role": "assistant", "content": "请提供订单号"},
            {"role": "user", "content": "O1001"},
        ],
        "session_facts": {
            "current_intent": "after_sales",
            "current_order_id": "O1001",
            "after_sales_reason": "鞋底开胶",
            "missing_slots": [],
        },
        "customer_memories": [
            {
                "memory_id": "M1",
                "title": "Shoe size preference",
                "description": "Usually wears size 42.",
                "confidence": "high",
            }
        ],
        "pending_confirmation": {
            "action_id": "A1",
            "action_type": "after_sales",
            "status": "pending_confirmation",
            "order_id": "O1001",
            "reason": "broken sole",
        },
    }
    config = {"configurable": {"thread_id": "conv-1"}}


class RecordingAgent:
    def __init__(self) -> None:
        self.payload = None
        self.config = None

    def invoke(self, payload, config=None):
        self.payload = payload
        self.config = config
        return {"messages": [AIMessage(content="子 Agent 已处理")]}


def test_pre_sales_tool_passes_original_message_and_subtask() -> None:
    agent = RecordingAgent()
    wrapped = make_pre_sales_tool(agent)

    result = wrapped.func("查询黑色 42 码库存", FakeRuntime())

    prompt = agent.payload["messages"][0]["content"]
    assert result == "子 Agent 已处理"
    assert "订单 O1001 鞋底开胶" in prompt
    assert "查询黑色 42 码库存" in prompt
    assert agent.config == FakeRuntime.config


def test_post_sales_tool_passes_original_message_and_subtask() -> None:
    agent = RecordingAgent()
    wrapped = make_post_sales_tool(agent)

    result = wrapped.func("创建售后申请", FakeRuntime())

    prompt = agent.payload["messages"][0]["content"]
    assert result == "子 Agent 已处理"
    assert "订单 O1001 鞋底开胶" in prompt
    assert "创建售后申请" in prompt
    assert agent.config == FakeRuntime.config


def test_post_sales_tool_injects_visual_evidence_context() -> None:
    agent = RecordingAgent()
    wrapped = make_post_sales_tool(agent)

    wrapped.func("创建售后申请", FakeRuntimeWithVisual())

    prompt = agent.payload["messages"][0]["content"]
    assert "图片证据上下文" in prompt
    assert "summary: 图片不清晰，无法确认鞋底问题" in prompt
    assert "confidence: 0.42" in prompt
    assert "usable_for_draft: false" in prompt
    assert "needs_clarification: true" in prompt
    assert "asset_key: conv-1/evidence.jpg" in prompt


def test_post_sales_tool_injects_summary_memories_and_pending_context() -> None:
    agent = RecordingAgent()
    wrapped = make_post_sales_tool(agent)

    wrapped.func("创建售后申请", FakeRuntimeWithCompactContext())

    prompt = agent.payload["messages"][0]["content"]
    assert "Conversation summary:" in prompt
    assert "User previously asked about order O1001." in prompt
    assert "Recent conversation:" in prompt
    assert "User: 我买的鞋开胶了，想售后" in prompt
    assert "Assistant: 请提供订单号" in prompt
    assert "Session facts:" in prompt
    assert "current_order_id: O1001" in prompt
    assert "after_sales_reason: 鞋底开胶" in prompt
    assert "Active customer memories:" in prompt
    assert "Shoe size preference" in prompt
    assert "Usually wears size 42." in prompt
    assert "Pending confirmation:" in prompt
    assert "A1" in prompt
