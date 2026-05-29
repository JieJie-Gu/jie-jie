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
    message: str
    has_image: bool
    visual_evidence: dict[str, Any] | None
    asset_key: str | None
    conversation_summary: str | None
    customer_memories: list[dict[str, Any]]
    tools_invoked: list[str]
    business_result: dict[str, Any] | None
    pending_confirmation: dict[str, Any] | None
    reply: str | None
    status: str
