from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from smart_cs.agents.state import RouteAnalysis, SupervisorDecision
from smart_cs.tools.executor import AuthorizedToolExecutor


@dataclass(frozen=True)
class SpecialistExecution:
    agents_invoked: list[str]
    results: list[dict[str, Any]]
    result: dict[str, Any]
    pending_confirmation: dict[str, Any] | None = None


class SpecialistDispatcher:
    """Execute an already validated supervisor plan through authorized tools."""

    def __init__(self, executor: AuthorizedToolExecutor) -> None:
        self.executor = executor
        self._registry: dict[
            str, Callable[[str, str, RouteAnalysis, str | None, str | None], dict[str, Any]]
        ] = {
            "ProductAgent": self._product,
            "OrderAgent": self._order,
            "KnowledgeAgent": self._knowledge,
            "AfterSalesAgent": self._after_sales,
            "HandoffAgent": self._handoff,
        }

    def execute(
        self,
        *,
        message: str,
        customer_id: str,
        route: RouteAnalysis,
        decision: SupervisorDecision,
        conversation_id: str | None = None,
        idempotency_key: str | None = None,
    ) -> SpecialistExecution:
        results: list[dict[str, Any]] = []
        for agent_name in decision.agents:
            results.append(
                self._registry[agent_name](
                    message, customer_id, route, conversation_id, idempotency_key
                )
            )

        result = results[-1]
        pending_confirmation = (
            result
            if decision.requires_confirmation and result.get("status") == "pending_confirmation"
            else None
        )
        return SpecialistExecution(
            agents_invoked=list(decision.agents),
            results=results,
            result=result,
            pending_confirmation=pending_confirmation,
        )

    def _product(
        self,
        message: str,
        _customer_id: str,
        _route: RouteAnalysis,
        _conversation_id: str | None,
        _idempotency_key: str | None,
    ) -> dict[str, Any]:
        return self.executor.invoke("search_products", {"query": message})

    def _order(
        self,
        _message: str,
        customer_id: str,
        route: RouteAnalysis,
        _conversation_id: str | None,
        _idempotency_key: str | None,
    ) -> dict[str, Any]:
        order_id = route.entities.get("order_id")
        if order_id is None:
            return {"status": "information_required", "message": "请提供需要查询的订单编号。"}
        return self.executor.invoke(
            "lookup_order", {"customer_id": customer_id, "order_id": order_id}
        )

    @staticmethod
    def _knowledge(
        _message: str,
        _customer_id: str,
        _route: RouteAnalysis,
        _conversation_id: str | None,
        _idempotency_key: str | None,
    ) -> dict[str, Any]:
        return {"status": "unavailable", "message": "知识库将在 RAG 阶段启用。"}

    def _after_sales(
        self,
        message: str,
        customer_id: str,
        route: RouteAnalysis,
        conversation_id: str | None,
        idempotency_key: str | None,
    ) -> dict[str, Any]:
        order_id = route.entities.get("order_id")
        if order_id is None:
            return {"status": "information_required", "message": "请提供需要售后的订单编号。"}
        arguments: dict[str, Any] = {
            "customer_id": customer_id,
            "order_id": order_id,
            "reason": message,
        }
        self._add_request_identity(arguments, conversation_id, idempotency_key)
        return self.executor.invoke(
            "draft_after_sales",
            arguments,
        )

    def _handoff(
        self,
        message: str,
        customer_id: str,
        _route: RouteAnalysis,
        conversation_id: str | None,
        idempotency_key: str | None,
    ) -> dict[str, Any]:
        arguments: dict[str, Any] = {"customer_id": customer_id, "reason": message}
        self._add_request_identity(arguments, conversation_id, idempotency_key)
        return self.executor.invoke(
            "draft_handoff", arguments
        )

    @staticmethod
    def _add_request_identity(
        arguments: dict[str, Any], conversation_id: str | None, idempotency_key: str | None
    ) -> None:
        if conversation_id is not None:
            arguments["conversation_id"] = conversation_id
        if idempotency_key is not None:
            arguments["idempotency_key"] = idempotency_key
