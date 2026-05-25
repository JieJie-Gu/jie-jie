from __future__ import annotations

from collections.abc import Callable
from time import perf_counter
from typing import Any

from smart_cs.domain.enums import ActionType, ToolCallStatus
from smart_cs.domain.errors import InvalidActionState, ToolPermissionError
from smart_cs.domain.models import Order, PendingAction, Product, Ticket
from smart_cs.domain.repositories import CustomerFactsRepository
from smart_cs.tools.customer_tools import CUSTOMER_TOOL_SCHEMAS


class AuthorizedToolExecutor:
    """Execute declared customer tools under deterministic business permissions."""

    def __init__(self, repository: CustomerFactsRepository) -> None:
        self.repository = repository
        self._handlers: dict[str, Callable[[dict[str, Any]], dict[str, Any]]] = {
            "search_products": self._search_products,
            "lookup_order": self._lookup_order,
            "draft_after_sales": self._draft_after_sales,
            "draft_handoff": self._draft_handoff,
        }
        self.declared_tools = {tool.name: tool for tool in CUSTOMER_TOOL_SCHEMAS}

    def invoke(self, tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        provided_arguments = dict(arguments)

        def operation() -> dict[str, Any]:
            if tool_name not in self.declared_tools:
                raise ValueError(f"Unknown customer tool: {tool_name}")
            return self._handlers[tool_name](provided_arguments)

        return self._audited_call(tool_name, provided_arguments, operation)

    def submit_confirmed_action(self, action_id: str, customer_id: str) -> dict[str, Any]:
        arguments = {"action_id": action_id, "customer_id": customer_id}

        def operation() -> dict[str, Any]:
            action, ticket = self.repository.submit_pending_action(action_id, customer_id)
            return self._action_result(action, ticket)

        return self._audited_call("submit_confirmed_action", arguments, operation)

    def cancel_pending_action(self, action_id: str, customer_id: str) -> dict[str, Any]:
        arguments = {"action_id": action_id, "customer_id": customer_id}

        def operation() -> dict[str, Any]:
            action = self.repository.cancel_pending_action(action_id, customer_id)
            return self._action_result(action)

        return self._audited_call("cancel_pending_action", arguments, operation)

    def _search_products(self, arguments: dict[str, Any]) -> dict[str, Any]:
        query = str(arguments["query"])
        return {"products": [self._product_result(product) for product in self.repository.search_products(query)]}

    def _lookup_order(self, arguments: dict[str, Any]) -> dict[str, Any]:
        customer_id = str(arguments["customer_id"])
        order_id = str(arguments["order_id"])
        order = self._owned_order(customer_id, order_id)
        return self._order_result(order)

    def _draft_after_sales(self, arguments: dict[str, Any]) -> dict[str, Any]:
        customer_id = str(arguments["customer_id"])
        order_id = str(arguments["order_id"])
        self._owned_order(customer_id, order_id)
        action = self.repository.create_pending_action(
            customer_id=customer_id,
            action_type=ActionType.AFTER_SALES.value,
            order_id=order_id,
            reason=str(arguments["reason"]),
        )
        return self._action_result(action)

    def _draft_handoff(self, arguments: dict[str, Any]) -> dict[str, Any]:
        customer_id = str(arguments["customer_id"])
        if not self.repository.customer_exists(customer_id):
            raise ToolPermissionError("Customer is not available for handoff")
        action = self.repository.create_pending_action(
            customer_id=customer_id,
            action_type=ActionType.HANDOFF.value,
            reason=str(arguments["reason"]),
        )
        return self._action_result(action)

    def _owned_order(self, customer_id: str, order_id: str) -> Order:
        order = self.repository.get_owned_order(customer_id, order_id)
        if order is None:
            raise ToolPermissionError("Order is not available to this customer")
        return order

    def _audited_call(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        operation: Callable[[], dict[str, Any]],
    ) -> dict[str, Any]:
        started = perf_counter()
        customer_id = arguments.get("customer_id")
        try:
            result = operation()
        except Exception as error:
            self.repository.record_tool_call(
                tool_name=tool_name,
                arguments=arguments,
                customer_id=str(customer_id) if customer_id is not None else None,
                status=ToolCallStatus.REJECTED.value,
                error_type=type(error).__name__,
                duration_ms=self._duration_ms(started),
            )
            raise
        self.repository.record_tool_call(
            tool_name=tool_name,
            arguments=arguments,
            customer_id=str(customer_id) if customer_id is not None else None,
            status=ToolCallStatus.SUCCEEDED.value,
            result=result,
            duration_ms=self._duration_ms(started),
        )
        return result

    @staticmethod
    def _duration_ms(started: float) -> int:
        return max(0, round((perf_counter() - started) * 1000))

    @staticmethod
    def _product_result(product: Product) -> dict[str, Any]:
        return {
            "product_id": product.id,
            "name": product.name,
            "description": product.description,
            "price_cents": product.price_cents,
        }

    @staticmethod
    def _order_result(order: Order) -> dict[str, Any]:
        return {
            "order_id": order.id,
            "customer_id": order.customer_id,
            "product_id": order.product_id,
            "status": order.status,
            "quantity": order.quantity,
            "total_cents": order.total_cents,
        }

    @staticmethod
    def _action_result(action: PendingAction, ticket: Ticket | None = None) -> dict[str, Any]:
        result: dict[str, Any] = {
            "action_id": action.id,
            "customer_id": action.customer_id,
            "action_type": action.action_type,
            "status": action.status,
            "reason": action.reason,
        }
        if action.order_id is not None:
            result["order_id"] = action.order_id
        if ticket is not None:
            result["ticket_id"] = ticket.id
        return result


__all__ = ["AuthorizedToolExecutor", "InvalidActionState", "ToolPermissionError"]
