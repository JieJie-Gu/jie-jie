from contextlib import AbstractContextManager
from typing import Any, Protocol

from smart_cs.domain.models import Conversation, Order, PendingAction, Product, Ticket, ToolCall


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

    def release_turn_lease(
        self, conversation_id: str, customer_id: str, token: str, *, session: Any | None = None
    ) -> None: ...

    def customer_exists(self, customer_id: str, *, session: Any | None = None) -> bool: ...

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
