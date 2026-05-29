from __future__ import annotations

from typing import Any

from langchain.tools import ToolRuntime, tool
from langchain_core.messages import BaseMessage


def make_pre_sales_tool(pre_sales_agent: Any):
    @tool
    def use_pre_sales_agent(request: str, runtime: ToolRuntime) -> str:
        """
        Handle product consulting, product recommendation, price, inventory,
        promotion, size guidance, and purchase questions.
        """

        prompt = _subagent_prompt(runtime, request, agent_label="售前")
        result = pre_sales_agent.invoke(
            {"messages": [{"role": "user", "content": prompt}]},
            config=runtime.config,
        )
        return _last_message_text(result)

    return use_pre_sales_agent


def make_post_sales_tool(post_sales_agent: Any):
    @tool
    def use_post_sales_agent(request: str, runtime: ToolRuntime) -> str:
        """
        Handle order queries, logistics, returns, refunds, exchanges,
        complaints, after-sales requests, and human handoff.
        """

        prompt = _subagent_prompt(
            runtime,
            request,
            agent_label="售后",
            include_visual_context=True,
        )
        result = post_sales_agent.invoke(
            {"messages": [{"role": "user", "content": prompt}]},
            config=runtime.config,
        )
        return _last_message_text(result)

    return use_post_sales_agent


def _subagent_prompt(
    runtime: ToolRuntime,
    request: str,
    *,
    agent_label: str,
    include_visual_context: bool = False,
) -> str:
    original = _latest_human_text(runtime.state.get("messages", []))
    visual_context = (
        _visual_context_block(runtime.state)
        if include_visual_context
        else ""
    )
    return (
        "你正在处理以下用户客服请求：\n\n"
        f"{original}\n\n"
        f"{visual_context}"
        f"你被分配的{agent_label}子任务是：\n\n"
        f"{request}"
    )


def _visual_context_block(state: dict[str, Any]) -> str:
    evidence = state.get("visual_evidence")
    if not isinstance(evidence, dict):
        return ""
    confidence = _float_or_zero(evidence.get("confidence"))
    needs_clarification = bool(evidence.get("needs_clarification"))
    usable_for_draft = confidence >= 0.8 and not needs_clarification
    summary = str(evidence.get("summary") or "")
    return (
        "图片证据上下文：\n"
        f"- summary: {summary}\n"
        f"- confidence: {confidence}\n"
        f"- usable_for_draft: {str(usable_for_draft).lower()}\n"
        f"- needs_clarification: {str(needs_clarification).lower()}\n"
        f"- asset_key: {state.get('asset_key') or ''}\n\n"
    )


def _float_or_zero(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _latest_human_text(messages: list[Any]) -> str:
    for message in reversed(messages):
        if getattr(message, "type", None) == "human":
            return _message_text(message)
        if isinstance(message, dict) and message.get("role") in {"user", "human"}:
            return str(message.get("content", ""))
    return ""


def _last_message_text(result: dict[str, Any]) -> str:
    messages = result.get("messages") or []
    if not messages:
        return ""
    return _message_text(messages[-1])


def _message_text(message: BaseMessage | Any) -> str:
    content = getattr(message, "content", message)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text") or item.get("content")
                if text:
                    parts.append(str(text))
            else:
                parts.append(str(item))
        return "\n".join(parts)
    return str(content)
