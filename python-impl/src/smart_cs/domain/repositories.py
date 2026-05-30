# 定义客服事实仓库的协议接口。

from contextlib import AbstractContextManager
from typing import Any, Protocol

from smart_cs.domain.models import (
    AgentRun,
    Conversation,
    MemoryRecord,
    Message,
    Order,
    PendingAction,
    Product,
    Ticket,
    ToolCall,
)


class CustomerFactsRepository(Protocol):
    """Business-specific persistence operations used by authorised tools."""

    def transaction(self) -> AbstractContextManager[Any]: ...

    def claim_conversation(
        self, conversation_id: str, customer_id: str, *, session: Any | None = None
    ) -> Conversation: ...

    def require_conversation_owner(
        self, conversation_id: str, customer_id: str, *, session: Any | None = None
    ) -> Conversation: ...

    def acquire_turn_lease(
        self,
        conversation_id: str,
        customer_id: str,
        token: str,
        *,
        ttl_seconds: float,
        session: Any | None = None,
    ) -> None: ...

    def renew_turn_lease(
        self,
        conversation_id: str,
        customer_id: str,
        token: str,
        *,
        ttl_seconds: float,
        session: Any | None = None,
    ) -> None: ...

    def require_active_turn_lease(
        self,
        conversation_id: str,
        customer_id: str,
        token: str,
        *,
        session: Any | None = None,
    ) -> None: ...

    def release_turn_lease(
        self, conversation_id: str, customer_id: str, token: str, *, session: Any | None = None
    ) -> None: ...

    def customer_exists(self, customer_id: str, *, session: Any | None = None) -> bool: ...

    def record_message(
        self,
        conversation_id: str,
        customer_id: str,
        role: str,
        content: str,
        content_type: str = "text",
        asset_key: str | None = None,
        visual_evidence: dict[str, Any] | None = None,
        session: Any | None = None,
    ) -> Message: ...

    def latest_message(self, conversation_id: str) -> Message | None: ...

    def list_recent_messages(
        self,
        conversation_id: str,
        customer_id: str,
        *,
        limit: int = 10,
    ) -> list[dict[str, Any]]: ...

    def get_memory(self, namespace: tuple[str, str, str], key: str) -> MemoryRecord | None: ...

    def list_memory_candidates(
        self,
        *,
        customer_id: str | None = None,
        status: str = "pending",
        limit: int = 50,
    ) -> list[dict[str, Any]]: ...

    def approve_memory_candidate(
        self,
        *,
        candidate_key: str,
        customer_id: str,
        reviewer_id: str,
        edited_value: dict[str, Any] | None = None,
    ) -> dict[str, Any]: ...

    def reject_memory_candidate(
        self,
        *,
        candidate_key: str,
        customer_id: str,
        reviewer_id: str,
        reason: str,
    ) -> dict[str, Any]: ...

    def search_products(self, query: str) -> list[Product]: ...

    def get_owned_order(self, customer_id: str, order_id: str, *, session: Any | None = None) -> Order | None: ...

    def create_pending_action(
        self,
        customer_id: str,
        action_type: str,
        reason: str,
        order_id: str | None = None,
        conversation_id: str | None = None,
        idempotency_key: str | None = None,
        *,
        session: Any | None = None,
    ) -> PendingAction: ...

    def get_pending_action(
        self, conversation_id: str, customer_id: str, *, session: Any | None = None
    ) -> PendingAction | None: ...

    def get_latest_action(
        self, conversation_id: str, customer_id: str, *, session: Any | None = None
    ) -> PendingAction | None: ...

    def get_action(
        self,
        conversation_id: str,
        customer_id: str,
        action_id: str,
        *,
        session: Any | None = None,
    ) -> PendingAction: ...

    def get_ticket_for_action(self, action_id: str, *, session: Any | None = None) -> Ticket | None: ...

    def submit_pending_action(
        self, action_id: str, customer_id: str, *, session: Any | None = None
    ) -> tuple[PendingAction, Ticket]: ...

    def cancel_pending_action(
        self, action_id: str, customer_id: str, *, session: Any | None = None
    ) -> PendingAction: ...

    def record_tool_call(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        status: str,
        customer_id: str | None = None,
        result: dict[str, Any] | None = None,
        error_type: str | None = None,
        duration_ms: int = 0,
        session: Any | None = None,
    ) -> ToolCall: ...

    def record_agent_run(
        self,
        conversation_id: str,
        customer_id: str,
        agents: list[str],
        status: str,
        pending_action_id: str | None = None,
        reply: str | None = None,
        session: Any | None = None,
    ) -> AgentRun: ...

    def update_agent_run_for_action(
        self,
        conversation_id: str,
        customer_id: str,
        pending_action_id: str,
        status: str,
        reply: str | None = None,
        agents: list[str] | None = None,
        session: Any | None = None,
    ) -> AgentRun | None: ...

    def list_agent_runs(self, conversation_id: str, customer_id: str) -> list[AgentRun]: ...
