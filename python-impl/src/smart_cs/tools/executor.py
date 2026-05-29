from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from time import perf_counter
from typing import Any

from smart_cs.domain.enums import ActionStatus, ActionType, ToolCallStatus
from smart_cs.domain.errors import InvalidActionState, ToolPermissionError
from smart_cs.domain.models import Order, PendingAction, Product, Ticket
from smart_cs.domain.repositories import CustomerFactsRepository
from smart_cs.tools.customer_tools import CUSTOMER_TOOL_SCHEMAS
from smart_cs.tools.policy import ToolPolicy, ToolRegistry, default_tool_registry


@dataclass(frozen=True)
class TurnFence:
    conversation_id: str
    lease_token: str


class AuthorizedToolExecutor:
    """Execute declared customer tools under deterministic business permissions."""

    def __init__(
        self,
        repository: CustomerFactsRepository,
        tool_registry: ToolRegistry | None = None,
    ) -> None:
        self.repository = repository
        self.tool_registry = tool_registry or default_tool_registry()
        self._read_handlers: dict[str, Callable[[dict[str, Any]], dict[str, Any]]] = {
            "search_products": self._search_products,
            "lookup_order": self._lookup_order,
        }
        self._write_handlers: dict[str, Callable[[dict[str, Any], Any], dict[str, Any]]] = {
            "draft_after_sales": self._draft_after_sales,
            "draft_handoff": self._draft_handoff,
        }
        self.declared_tools = {tool.name: tool for tool in CUSTOMER_TOOL_SCHEMAS}

    def invoke(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        *,
        caller_agent: str,
        turn_fence: TurnFence | None = None,
    ) -> dict[str, Any]:
        provided_arguments = dict(arguments)
        policy = self._authorize_tool(tool_name, caller_agent)
        if tool_name in self._write_handlers and not policy.requires_confirmation:
            raise ToolPermissionError(f"Write tool {tool_name} must require confirmation")

        def operation() -> dict[str, Any]:
            if tool_name not in self.declared_tools:
                raise ValueError(f"Unknown customer tool: {tool_name}")
            return self._read_handlers[tool_name](provided_arguments)

        if tool_name in self._write_handlers:
            return self._audited_write_call(
                tool_name,
                provided_arguments,
                lambda session: self._invoke_fenced_write(
                    self._write_handlers[tool_name],
                    provided_arguments,
                    turn_fence,
                    session,
                ),
            )
        return self._audited_call(tool_name, provided_arguments, operation)

    def claim_conversation(self, conversation_id: str, customer_id: str) -> None:
        arguments = {"conversation_id": conversation_id, "customer_id": customer_id}

        def operation(session: Any) -> dict[str, Any]:
            conversation = self.repository.claim_conversation(
                conversation_id, customer_id, session=session
            )
            return {"conversation_id": conversation.id, "customer_id": conversation.customer_id}

        self._audited_write_call("claim_conversation", arguments, operation)

    def require_conversation_owner(self, conversation_id: str, customer_id: str) -> None:
        self.repository.require_conversation_owner(conversation_id, customer_id)

    def acquire_turn_lease(
        self, conversation_id: str, customer_id: str, token: str, *, ttl_seconds: float
    ) -> None:
        self.repository.acquire_turn_lease(
            conversation_id, customer_id, token, ttl_seconds=ttl_seconds
        )

    def renew_turn_lease(
        self, conversation_id: str, customer_id: str, token: str, *, ttl_seconds: float
    ) -> None:
        self.repository.renew_turn_lease(
            conversation_id, customer_id, token, ttl_seconds=ttl_seconds
        )

    def release_turn_lease(self, conversation_id: str, customer_id: str, token: str) -> None:
        self.repository.release_turn_lease(conversation_id, customer_id, token)

    def pending_action_for_conversation(
        self, conversation_id: str, customer_id: str
    ) -> dict[str, Any] | None:
        action = self.repository.get_pending_action(conversation_id, customer_id)
        return self._action_result(action) if action is not None else None

    def latest_action_for_conversation(
        self, conversation_id: str, customer_id: str
    ) -> dict[str, Any] | None:
        action = self.repository.get_latest_action(conversation_id, customer_id)
        if action is None:
            return None
        return self._persisted_action_result(action)

    def action_for_conversation(
        self, conversation_id: str, customer_id: str, action_id: str
    ) -> dict[str, Any]:
        action = self.repository.get_action(conversation_id, customer_id, action_id)
        return self._persisted_action_result(action)

    def submit_confirmed_action(
        self,
        action_id: str,
        customer_id: str,
        *,
        caller_agent: str,
        turn_fence: TurnFence | None = None,
    ) -> dict[str, Any]:
        self._authorize_tool("submit_confirmed_action", caller_agent)
        arguments = {"action_id": action_id, "customer_id": customer_id}

        def operation(session: Any) -> dict[str, Any]:
            self._require_write_fence(turn_fence, customer_id, session)
            action, ticket = self.repository.submit_pending_action(action_id, customer_id, session=session)
            return self._action_result(action, ticket)

        return self._audited_write_call("submit_confirmed_action", arguments, operation)

    def cancel_pending_action(
        self,
        action_id: str,
        customer_id: str,
        *,
        caller_agent: str,
        turn_fence: TurnFence | None = None,
    ) -> dict[str, Any]:
        self._authorize_tool("cancel_pending_action", caller_agent)
        arguments = {"action_id": action_id, "customer_id": customer_id}

        def operation(session: Any) -> dict[str, Any]:
            self._require_write_fence(turn_fence, customer_id, session)
            action = self.repository.cancel_pending_action(action_id, customer_id, session=session)
            return self._action_result(action)

        return self._audited_write_call("cancel_pending_action", arguments, operation)

    def _search_products(self, arguments: dict[str, Any]) -> dict[str, Any]:
        query = str(arguments["query"])
        return {"products": [self._product_result(product) for product in self.repository.search_products(query)]}

    def _lookup_order(self, arguments: dict[str, Any]) -> dict[str, Any]:
        customer_id = str(arguments["customer_id"])
        order_id = str(arguments["order_id"])
        order = self._owned_order(customer_id, order_id)
        return self._order_result(order)

    def _draft_after_sales(self, arguments: dict[str, Any], session: Any) -> dict[str, Any]:
        customer_id = str(arguments["customer_id"])
        order_id = str(arguments["order_id"])
        conversation_id = arguments.get("conversation_id")
        if conversation_id is not None:
            self.repository.require_conversation_owner(str(conversation_id), customer_id, session=session)
        self._owned_order(customer_id, order_id, session=session)
        action = self.repository.create_pending_action(
            customer_id=customer_id,
            action_type=ActionType.AFTER_SALES.value,
            order_id=order_id,
            reason=str(arguments["reason"]),
            conversation_id=str(conversation_id) if conversation_id is not None else None,
            idempotency_key=arguments.get("idempotency_key"),
            session=session,
        )
        return self._draft_result(action, arguments, session=session)

    def _draft_handoff(self, arguments: dict[str, Any], session: Any) -> dict[str, Any]:
        customer_id = str(arguments["customer_id"])
        conversation_id = arguments.get("conversation_id")
        if conversation_id is not None:
            self.repository.require_conversation_owner(str(conversation_id), customer_id, session=session)
        if not self.repository.customer_exists(customer_id, session=session):
            raise ToolPermissionError("Customer is not available for handoff")
        action = self.repository.create_pending_action(
            customer_id=customer_id,
            action_type=ActionType.HANDOFF.value,
            reason=str(arguments["reason"]),
            conversation_id=str(conversation_id) if conversation_id is not None else None,
            idempotency_key=arguments.get("idempotency_key"),
            session=session,
        )
        return self._draft_result(action, arguments, session=session)

    def _owned_order(self, customer_id: str, order_id: str, session: Any | None = None) -> Order:
        order = self.repository.get_owned_order(customer_id, order_id, session=session)
        if order is None:
            raise ToolPermissionError("Order is not available to this customer")
        return order

    def _authorize_tool(self, tool_name: str, caller_agent: str) -> ToolPolicy:
        policy = self.tool_registry.get(tool_name)
        if caller_agent not in policy.allowed_agents:
            raise ToolPermissionError(f"Tool {tool_name} is not allowed for {caller_agent}")
        return policy

    def _invoke_fenced_write(
        self,
        handler: Callable[[dict[str, Any], Any], dict[str, Any]],
        arguments: dict[str, Any],
        turn_fence: TurnFence | None,
        session: Any,
    ) -> dict[str, Any]:
        if (
            turn_fence is not None
            and arguments.get("conversation_id") != turn_fence.conversation_id
        ):
            raise ToolPermissionError("Turn fence does not match the conversation action")
        self._require_write_fence(turn_fence, str(arguments["customer_id"]), session)
        return handler(arguments, session)

    def _require_write_fence(
        self, turn_fence: TurnFence | None, customer_id: str, session: Any
    ) -> None:
        if turn_fence is None:
            return
        self.repository.require_active_turn_lease(
            turn_fence.conversation_id,
            customer_id,
            turn_fence.lease_token,
            session=session,
        )

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

    def _audited_write_call(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        operation: Callable[[Any], dict[str, Any]],
    ) -> dict[str, Any]:
        started = perf_counter()
        customer_id = arguments.get("customer_id")
        try:
            with self.repository.transaction() as session:
                result = operation(session)
                self.repository.record_tool_call(
                    tool_name=tool_name,
                    arguments=arguments,
                    customer_id=str(customer_id) if customer_id is not None else None,
                    status=ToolCallStatus.SUCCEEDED.value,
                    result=result,
                    duration_ms=self._duration_ms(started),
                    session=session,
                )
                return result
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

    def _persisted_action_result(
        self, action: PendingAction, *, session: Any | None = None
    ) -> dict[str, Any]:
        ticket = None
        if action.status == ActionStatus.SUBMITTED.value:
            ticket = self.repository.get_ticket_for_action(action.id, session=session)
        return self._action_result(action, ticket)

    def _draft_result(
        self, action: PendingAction, arguments: dict[str, Any], *, session: Any
    ) -> dict[str, Any]:
        result = self._persisted_action_result(action, session=session)
        requested_key = arguments.get("idempotency_key")
        if (
            arguments.get("conversation_id") is not None
            and requested_key is not None
            and action.idempotency_key != requested_key
        ):
            result["_canonical_pending"] = True
        return result

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


__all__ = ["AuthorizedToolExecutor", "InvalidActionState", "ToolPermissionError", "TurnFence"]
