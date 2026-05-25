from contextlib import AbstractContextManager
from typing import Any, Protocol

from smart_cs.domain.models import Order, PendingAction, Product, Ticket, ToolCall


class CustomerFactsRepository(Protocol):
    """Business-specific persistence operations used by authorised tools."""

    def transaction(self) -> AbstractContextManager[Any]: ...

    def customer_exists(self, customer_id: str, *, session: Any | None = None) -> bool: ...

    def search_products(self, query: str) -> list[Product]: ...

    def get_owned_order(self, customer_id: str, order_id: str, *, session: Any | None = None) -> Order | None: ...

    def create_pending_action(
        self,
        customer_id: str,
        action_type: str,
        reason: str,
        order_id: str | None = None,
        *,
        session: Any | None = None,
    ) -> PendingAction: ...

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
