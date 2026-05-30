# 定义 HTTP 请求和响应的 Pydantic schema。

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class ConversationCreateRequest(BaseModel):
    customer_id: str = Field(min_length=1)


class ConversationResponse(BaseModel):
    id: str
    customer_id: str


class MessageRequest(BaseModel):
    customer_id: str = Field(min_length=1)
    content: str = Field(min_length=1)


class ConfirmRequest(BaseModel):
    customer_id: str = Field(min_length=1)
    action_id: str = Field(min_length=1)
    approved: bool


class ConversationWorkflowResponse(BaseModel):
    status: str
    reply: str
    result: dict[str, Any] | None = None
    agents_invoked: list[str] = Field(default_factory=list)
    pending_action: dict[str, Any] | None = None
    visual_evidence: dict[str, Any] | None = None
    asset_key: str | None = None


class ToolCallItem(BaseModel):
    id: int
    tool_name: str
    customer_id: str | None
    arguments: dict[str, Any]
    result: dict[str, Any] | None
    status: str
    error_type: str | None
    duration_ms: int
    created_at: datetime


class ToolCallsResponse(BaseModel):
    tool_calls: list[ToolCallItem]


class AgentRunItem(BaseModel):
    id: str
    conversation_id: str
    agents: list[str]
    status: str
    pending_action_id: str | None
    reply: str | None
    created_at: datetime


class AgentRunsResponse(BaseModel):
    runs: list[AgentRunItem]
    tool_calls: list[ToolCallItem]
