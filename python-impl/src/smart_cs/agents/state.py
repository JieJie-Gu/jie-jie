from __future__ import annotations

from typing import Annotated, Any, Literal, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field


class RouteAnalysis(BaseModel):
    intent: Literal["product", "order", "knowledge", "after_sales", "handoff"]
    entities: dict[str, str] = Field(default_factory=dict)
    risk: Literal["low", "medium", "high"] = "low"
    confidence: Literal["low", "medium", "high"] = "medium"
    turn_type: Literal[
        "new_request",
        "follow_up",
        "correction",
        "confirmation_like",
        "rejection_like",
        "information_update",
    ] = "new_request"
    missing_entities: list[str] = Field(default_factory=list)
    escalation_signals: list[str] = Field(default_factory=list)
    referenced_memory_ids: list[str] = Field(default_factory=list)


class SupervisorDecision(BaseModel):
    agents: list[
        Literal[
            "ProductAgent",
            "OrderAgent",
            "KnowledgeAgent",
            "VisionAgent",
            "AfterSalesAgent",
            "HandoffAgent",
        ]
    ]
    action: Literal["read", "draft_after_sales", "draft_handoff"]
    requires_confirmation: bool = False
    missing_entities: list[str] = Field(default_factory=list)
    planning_flags: list[str] = Field(default_factory=list)
    handoff_reason: str | None = None
    referenced_memory_ids: list[str] = Field(default_factory=list)


class ConversationSlots(BaseModel):
    active_order_id: str | None = None
    active_product_id: str | None = None
    active_after_sales_id: str | None = None
    active_ticket_id: str | None = None
    last_intent: str | None = None
    last_entities: dict[str, str] = Field(default_factory=dict)
    unresolved_question: str | None = None
    last_tool_results: dict[str, Any] = Field(default_factory=dict)
    pending_action: dict[str, Any] | None = None
    action_status: str | None = None


class MemoryView(BaseModel):
    memory_id: str
    memory_type: str
    value: dict[str, Any] = Field(default_factory=dict)
    title: str | None = None
    description: str | None = None
    confidence: Literal["low", "medium", "high"] = "medium"
    source: str | None = None


class RouterContext(BaseModel):
    current_message: str
    recent_messages: list[dict[str, str]] = Field(default_factory=list)
    conversation_summary: str | None = None
    conversation_slots: ConversationSlots = Field(default_factory=ConversationSlots)
    pending_action: dict[str, Any] | None = None
    customer_memories: list[MemoryView] = Field(default_factory=list)
    has_image: bool = False
    visual_evidence: dict[str, Any] | None = None


class SupervisorContext(BaseModel):
    current_message: str
    route: RouteAnalysis
    recent_messages: list[dict[str, str]] = Field(default_factory=list)
    conversation_summary: str | None = None
    conversation_slots: ConversationSlots = Field(default_factory=ConversationSlots)
    pending_action: dict[str, Any] | None = None
    customer_memories: list[MemoryView] = Field(default_factory=list)
    has_image: bool = False
    visual_evidence: dict[str, Any] | None = None
    agent_capabilities: dict[str, str] = Field(default_factory=dict)
    tool_policies: list[dict[str, object]] = Field(default_factory=list)
    planning_constraints: list[str] = Field(default_factory=list)


class RuntimeContext(BaseModel):
    conversation_id: str
    customer_id: str
    prompt_version: str


class RuntimeState(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], add_messages]
    conversation_id: str
    customer_id: str
    request_id: str
    message: str
    has_image: bool
    visual_evidence: dict[str, Any] | None
    asset_key: str | None
    route: dict[str, Any]
    decision: dict[str, Any]
    decision_context: dict[str, Any]
    conversation_slots: dict[str, Any]
    conversation_summary: str | None
    customer_memories: list[dict[str, Any]]
    agents_invoked: list[str]
    specialist_results: list[dict[str, Any]]
    read_results: list[dict[str, Any]]
    policy_decision: dict[str, Any] | None
    business_result: dict[str, Any] | None
    pending_confirmation: dict[str, Any] | None
    guarded_contents: list[str]
    reply: str | None
    status: str
