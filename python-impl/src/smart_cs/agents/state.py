from __future__ import annotations

from typing import Any, Literal, TypedDict

from pydantic import BaseModel, Field


class RouteAnalysis(BaseModel):
    intent: Literal["product", "order", "knowledge", "after_sales", "handoff"]
    entities: dict[str, str] = Field(default_factory=dict)
    risk: Literal["low", "medium", "high"] = "low"


class SupervisorDecision(BaseModel):
    agents: list[
        Literal["ProductAgent", "OrderAgent", "KnowledgeAgent", "AfterSalesAgent", "HandoffAgent"]
    ]
    action: Literal["read", "draft_after_sales", "draft_handoff"]
    requires_confirmation: bool = False


class RuntimeState(TypedDict, total=False):
    conversation_id: str
    customer_id: str
    message: str
    route: dict[str, Any]
    decision: dict[str, Any]
    agents_invoked: list[str]
    specialist_results: list[dict[str, Any]]
    business_result: dict[str, Any] | None
    pending_confirmation: dict[str, Any] | None
    reply: str | None
    status: str
