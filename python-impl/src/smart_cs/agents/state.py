# 定义 supervisor graph 的共享运行时状态。

from __future__ import annotations

from typing import Annotated, Any, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages


class RuntimeState(TypedDict, total=False):
    """Minimal runtime state for the sub-agent-as-tool supervisor."""

    messages: Annotated[list[AnyMessage], add_messages]
    conversation_id: str
    customer_id: str
    request_id: str
    recent_messages: list[dict[str, Any]]
    session_facts: dict[str, Any]
    conversation_summary: str | None
    customer_memories: list[dict[str, Any]]
    pending_confirmation: dict[str, Any] | None
    visual_evidence: dict[str, Any] | None
    asset_key: str | None
    tools_invoked: list[str]
    business_result: dict[str, Any] | None
    reply: str | None
    status: str


def latest_human_text(messages: list[Any]) -> str:
    """从 LangGraph messages 中派生最近一条用户文本，避免维护第二个 message 状态源。"""

    for message in reversed(messages):
        if getattr(message, "type", None) == "human":
            return _message_text(message)
        if isinstance(message, dict) and message.get("role") in {"user", "human"}:
            return str(message.get("content", ""))
    return ""


def _message_text(message: AnyMessage | Any) -> str:
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
