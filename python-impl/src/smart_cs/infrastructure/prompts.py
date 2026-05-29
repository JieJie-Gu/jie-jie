from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate


PROMPT_VERSION = "prompt-context-memory-p0-v1"

ROUTER_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "你是客服路由分析器。只输出结构化 RouteAnalysis；不要选择工具，不要批准动作。",
        ),
        (
            "human",
            "请基于以下上下文分析当前消息。\n\n上下文：{context_json}",
        ),
    ]
)

SUPERVISOR_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "你是客服主管规划器。只输出结构化 SupervisorDecision；写动作必须 requires_confirmation=True。",
        ),
        (
            "human",
            "请基于以下上下文规划 specialist 执行顺序。\n\n上下文：{context_json}",
        ),
    ]
)
