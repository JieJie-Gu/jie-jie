from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from smart_cs.agents.knowledge import KnowledgeAgent
from smart_cs.agents.state import RouteAnalysis, SupervisorDecision
from smart_cs.tools.executor import AuthorizedToolExecutor, TurnFence


READ_AGENTS = {"ProductAgent", "OrderAgent", "KnowledgeAgent", "VisionAgent"}
WRITE_AGENTS = {"AfterSalesAgent", "HandoffAgent"}


@dataclass(frozen=True)
class SpecialistExecution:
    agents_invoked: list[str]
    results: list[dict[str, Any]]
    result: dict[str, Any]
    pending_confirmation: dict[str, Any] | None = None


class SpecialistDispatcher:
    """Execute an already validated supervisor plan through authorized tools."""

    def __init__(
        self, executor: AuthorizedToolExecutor, knowledge_agent: KnowledgeAgent | None = None
    ) -> None:
        self.executor = executor
        self.knowledge_agent = knowledge_agent
        self._registry: dict[str, Callable[..., dict[str, Any]]] = {
            "ProductAgent": self._product,
            "OrderAgent": self._order,
            "KnowledgeAgent": self._knowledge,
            "VisionAgent": self._vision,
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
        turn_fence: TurnFence | None = None,
        visual_evidence: dict[str, Any] | None = None,
        asset_key: str | None = None,
    ) -> SpecialistExecution:
        results: list[dict[str, Any]] = []
        for agent_name in decision.agents:
            results.append(
                self._registry[agent_name](
                    message,
                    customer_id,
                    route,
                    conversation_id,
                    idempotency_key,
                    turn_fence,
                    visual_evidence,
                    asset_key,
                )
            )

        result = dict(results[-1])
        canonical_pending = result.pop("_canonical_pending", False)
        if canonical_pending:
            results = [result]
            agents_invoked: list[str] = []
        else:
            results[-1] = result
            agents_invoked = list(decision.agents)
        pending_confirmation = (
            result
            if decision.requires_confirmation and result.get("status") == "pending_confirmation"
            else None
        )
        return SpecialistExecution(
            agents_invoked=agents_invoked,
            results=results,
            result=result,
            pending_confirmation=pending_confirmation,
        )

    def execute_read_agents(self, **kwargs: Any) -> SpecialistExecution:
        decision = kwargs["decision"]
        read_agents = [agent for agent in decision.agents if agent in READ_AGENTS]
        if not read_agents:
            return SpecialistExecution(
                agents_invoked=[],
                results=[],
                result={"status": "no_read"},
                pending_confirmation=None,
            )
        read_decision = decision.model_copy(update={"agents": read_agents, "action": "read"})
        return self.execute(**{**kwargs, "decision": read_decision})

    def execute_write_agents(self, **kwargs: Any) -> SpecialistExecution:
        decision = kwargs["decision"]
        write_agents = [agent for agent in decision.agents if agent in WRITE_AGENTS]
        if not write_agents:
            return SpecialistExecution(
                agents_invoked=[],
                results=[],
                result={"status": "no_write"},
                pending_confirmation=None,
            )
        write_decision = decision.model_copy(update={"agents": write_agents})
        return self.execute(**{**kwargs, "decision": write_decision})

    def _product(
        self,
        message: str,
        _customer_id: str,
        _route: RouteAnalysis,
        conversation_id: str | None,
        _idempotency_key: str | None,
        _turn_fence: TurnFence | None,
        _visual_evidence: dict[str, Any] | None,
        _asset_key: str | None,
    ) -> dict[str, Any]:
        arguments = {"query": message}
        if conversation_id is not None:
            arguments["conversation_id"] = conversation_id
        return self.executor.invoke("search_products", arguments, caller_agent="ProductAgent")

    def _order(
        self,
        _message: str,
        customer_id: str,
        route: RouteAnalysis,
        conversation_id: str | None,
        _idempotency_key: str | None,
        _turn_fence: TurnFence | None,
        _visual_evidence: dict[str, Any] | None,
        _asset_key: str | None,
    ) -> dict[str, Any]:
        order_id = route.entities.get("order_id")
        if order_id is None:
            return {"status": "information_required", "message": "请提供需要查询的订单编号。"}
        arguments = {"customer_id": customer_id, "order_id": order_id}
        if conversation_id is not None:
            arguments["conversation_id"] = conversation_id
        return self.executor.invoke("lookup_order", arguments, caller_agent="OrderAgent")

    def _knowledge(
        self,
        message: str,
        _customer_id: str,
        _route: RouteAnalysis,
        _conversation_id: str | None,
        _idempotency_key: str | None,
        _turn_fence: TurnFence | None,
        _visual_evidence: dict[str, Any] | None,
        _asset_key: str | None,
    ) -> dict[str, Any]:
        if self.knowledge_agent is not None:
            return self.knowledge_agent.answer(message).as_result()
        return {"status": "unavailable", "message": "知识库暂未配置，无法提供政策引用。"}

    def _vision(
        self,
        _message: str,
        _customer_id: str,
        _route: RouteAnalysis,
        _conversation_id: str | None,
        _idempotency_key: str | None,
        _turn_fence: TurnFence | None,
        visual_evidence: dict[str, Any] | None,
        asset_key: str | None,
    ) -> dict[str, Any]:
        return {
            "status": "visual_evidence",
            "visual_evidence": visual_evidence or {},
            "asset_key": asset_key,
        }

    def _after_sales(
        self,
        message: str,
        customer_id: str,
        route: RouteAnalysis,
        conversation_id: str | None,
        idempotency_key: str | None,
        turn_fence: TurnFence | None,
        _visual_evidence: dict[str, Any] | None,
        _asset_key: str | None,
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
            caller_agent="AfterSalesAgent",
            turn_fence=turn_fence,
        )

    def _handoff(
        self,
        message: str,
        customer_id: str,
        _route: RouteAnalysis,
        conversation_id: str | None,
        idempotency_key: str | None,
        turn_fence: TurnFence | None,
        _visual_evidence: dict[str, Any] | None,
        _asset_key: str | None,
    ) -> dict[str, Any]:
        arguments: dict[str, Any] = {"customer_id": customer_id, "reason": message}
        self._add_request_identity(arguments, conversation_id, idempotency_key)
        return self.executor.invoke(
            "draft_handoff",
            arguments,
            caller_agent="HandoffAgent",
            turn_fence=turn_fence,
        )

    @staticmethod
    def _add_request_identity(
        arguments: dict[str, Any], conversation_id: str | None, idempotency_key: str | None
    ) -> None:
        if conversation_id is not None:
            arguments["conversation_id"] = conversation_id
        if idempotency_key is not None:
            arguments["idempotency_key"] = idempotency_key
