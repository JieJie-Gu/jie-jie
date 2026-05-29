from __future__ import annotations

from typing import Any

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage
from langchain_core.messages.utils import count_tokens_approximately, trim_messages

from smart_cs.agents.state import (
    ConversationSlots,
    MemoryView,
    RouteAnalysis,
    RouterContext,
    SupervisorContext,
)
from smart_cs.tools.policy import default_tool_policy_view


AGENT_CAPABILITIES = {
    "ProductAgent": "Read-only product search and product facts.",
    "OrderAgent": "Read-only order and logistics lookup for the current customer.",
    "KnowledgeAgent": "Read-only policy retrieval with citations.",
    "VisionAgent": "Read-only visual evidence projection for after-sales images.",
    "AfterSalesAgent": "Creates after-sales draft only after policy allows it.",
    "HandoffAgent": "Creates human handoff draft when handoff policy requires it.",
}


class ContextProjector:
    def __init__(self, *, max_context_tokens: int = 2048) -> None:
        self.max_context_tokens = max_context_tokens

    def build_router_context(self, state: dict[str, Any]) -> RouterContext:
        return RouterContext(
            current_message=str(state["message"]),
            recent_messages=self._recent_messages(state),
            conversation_summary=state.get("conversation_summary"),
            conversation_slots=self._slots(state),
            pending_action=state.get("pending_confirmation"),
            customer_memories=self._memory_views(state),
            has_image=bool(state.get("has_image")),
            visual_evidence=state.get("visual_evidence"),
        )

    def build_supervisor_context(
        self, state: dict[str, Any], route: RouteAnalysis
    ) -> SupervisorContext:
        return SupervisorContext(
            current_message=str(state["message"]),
            route=route,
            recent_messages=self._recent_messages(state),
            conversation_summary=state.get("conversation_summary"),
            conversation_slots=self._slots(state),
            pending_action=state.get("pending_confirmation"),
            customer_memories=self._memory_views(state),
            has_image=bool(state.get("has_image")),
            visual_evidence=state.get("visual_evidence"),
            agent_capabilities=AGENT_CAPABILITIES,
            tool_policies=default_tool_policy_view(),
            planning_constraints=[
                "写动作必须 requires_confirmation=True",
                "draft_after_sales 必须包含 OrderAgent 和 KnowledgeAgent，且 AfterSalesAgent 位于最后",
                "draft_handoff 必须由 HandoffAgent 位于最后",
            ],
        )

    def _recent_messages(self, state: dict[str, Any]) -> list[dict[str, str]]:
        messages = list(state.get("messages") or [])
        if not messages:
            return []
        trimmed = trim_messages(
            messages,
            strategy="last",
            token_counter=count_tokens_approximately,
            max_tokens=self.max_context_tokens,
            start_on="human",
            end_on=("human", "tool"),
        )
        return [self._message_view(message) for message in trimmed]

    @staticmethod
    def _message_view(message: AnyMessage) -> dict[str, str]:
        if isinstance(message, HumanMessage):
            role = "human"
        elif isinstance(message, AIMessage):
            role = "ai"
        elif isinstance(message, SystemMessage):
            role = "system"
        else:
            role = getattr(message, "type", "message")
        return {"role": role, "content": str(message.content)}

    @staticmethod
    def _slots(state: dict[str, Any]) -> ConversationSlots:
        return ConversationSlots.model_validate(state.get("conversation_slots") or {})

    @staticmethod
    def _memory_views(state: dict[str, Any]) -> list[MemoryView]:
        return [MemoryView.model_validate(item) for item in state.get("customer_memories") or []]
