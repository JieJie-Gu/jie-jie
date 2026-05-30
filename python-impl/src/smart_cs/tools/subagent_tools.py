# 将售前和售后子 Agent 包装成 supervisor 可调用的高级工具。

from __future__ import annotations

from typing import Any

from langchain.tools import ToolRuntime, tool
from langchain_core.messages import BaseMessage

from smart_cs.agents.state import latest_human_text


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
    original = latest_human_text(runtime.state.get("messages", []))
    compact_context = _compact_context_block(runtime.state)
    visual_context = (
        _visual_context_block(runtime.state)
        if include_visual_context
        else ""
    )
    return (
        "你正在处理以下用户客服请求：\n\n"
        f"{original}\n\n"
        f"{compact_context}"
        f"{visual_context}"
        f"你被分配的{agent_label}子任务是：\n\n"
        f"{request}"
    )


def _compact_context_block(state: dict[str, Any]) -> str:
    blocks: list[str] = []
    summary = state.get("conversation_summary")
    if summary:
        blocks.append("Conversation summary:\n" + str(summary))

    recent_messages = list(state.get("recent_messages") or [])
    if recent_messages:
        lines = ["Recent conversation:"]
        role_names = {"user": "User", "human": "User", "assistant": "Assistant", "ai": "Assistant"}
        for message in recent_messages:
            if not isinstance(message, dict):
                continue
            role = role_names.get(str(message.get("role") or "").lower(), str(message.get("role") or "Message"))
            lines.append(f"- {role}: {message.get('content') or ''}")
        if len(lines) > 1:
            blocks.append("\n".join(lines))

    session_facts = state.get("session_facts")
    if isinstance(session_facts, dict) and any(
        value not in (None, "", [], {}) for value in session_facts.values()
    ):
        lines = ["Session facts:"]
        for key, value in session_facts.items():
            if value in (None, "", [], {}):
                continue
            lines.append(f"- {key}: {value}")
        blocks.append("\n".join(lines))

    memories = list(state.get("customer_memories") or [])[:5]
    if memories:
        lines = ["Active customer memories:"]
        for memory in memories:
            if not isinstance(memory, dict):
                continue
            memory_id = memory.get("memory_id") or memory.get("key") or ""
            title = memory.get("title") or ""
            description = memory.get("description") or ""
            confidence = memory.get("confidence") or ""
            lines.append(
                f"- {memory_id}: {title}; {description}; confidence={confidence}"
            )
        if len(lines) > 1:
            blocks.append("\n".join(lines))

    pending = state.get("pending_confirmation")
    if isinstance(pending, dict) and pending:
        blocks.append(
            "Pending confirmation:\n"
            f"- action_id: {pending.get('action_id') or ''}\n"
            f"- action_type: {pending.get('action_type') or ''}\n"
            f"- status: {pending.get('status') or ''}\n"
            f"- order_id: {pending.get('order_id') or ''}\n"
            f"- reason: {pending.get('reason') or ''}"
        )

    return ("\n\n".join(blocks) + "\n\n") if blocks else ""


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
