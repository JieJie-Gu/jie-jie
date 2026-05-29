from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage

from smart_cs.tools.subagent_tools import make_post_sales_tool, make_pre_sales_tool


class FakeRuntime:
    state = {"messages": [HumanMessage(content="订单 O1001 鞋底开胶，帮我申请售后")]}
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
