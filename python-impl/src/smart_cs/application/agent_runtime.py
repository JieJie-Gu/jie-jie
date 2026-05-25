from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any, Protocol

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from smart_cs.agents.guardrails import ResponseGuard
from smart_cs.agents.router import RouterAgent, RoutingDecisionModel
from smart_cs.agents.specialists import SpecialistDispatcher
from smart_cs.agents.state import RouteAnalysis, RuntimeState, SupervisorDecision
from smart_cs.agents.supervisor import PlanningDecisionModel, SupervisorAgent
from smart_cs.domain.errors import ToolPermissionError
from smart_cs.tools.executor import AuthorizedToolExecutor


class DecisionModel(RoutingDecisionModel, PlanningDecisionModel, Protocol):
    pass


class AgentRuntime:
    """Run customer service orchestration with durable confirmation pauses."""

    def __init__(
        self,
        *,
        executor: AuthorizedToolExecutor,
        decision_model: DecisionModel,
        checkpoint_path: str | Path,
    ) -> None:
        self.executor = executor
        self.router = RouterAgent(decision_model)
        self.supervisor = SupervisorAgent(decision_model)
        self.specialists = SpecialistDispatcher(executor)
        self.guard = ResponseGuard()

        checkpoint_file = Path(checkpoint_path)
        checkpoint_file.parent.mkdir(parents=True, exist_ok=True)
        self._checkpoint_connection = sqlite3.connect(str(checkpoint_file), check_same_thread=False)
        self._checkpointer = SqliteSaver(self._checkpoint_connection)
        self.graph = self._build_graph()

    def invoke(self, conversation_id: str, customer_id: str, message: str) -> dict[str, Any]:
        result = self.graph.invoke(
            {
                "conversation_id": conversation_id,
                "customer_id": customer_id,
                "message": message,
                "route": {},
                "decision": {},
                "agents_invoked": [],
                "specialist_results": [],
                "business_result": None,
                "pending_confirmation": None,
                "reply": None,
                "status": "running",
            },
            config=self._config(conversation_id),
        )
        return self._public_result(result)

    def confirm(self, conversation_id: str, customer_id: str, *, approved: bool) -> dict[str, Any]:
        config = self._config(conversation_id)
        state = self.graph.get_state(config).values
        if state.get("customer_id") != customer_id:
            raise ToolPermissionError("Conversation is not available to this customer")
        if state.get("pending_confirmation") is None:
            raise ValueError("Conversation has no pending confirmation")

        result = self.graph.invoke(Command(resume={"approved": approved}), config=config)
        return self._public_result(result)

    def close(self) -> None:
        self._checkpoint_connection.close()

    def _build_graph(self):
        workflow = StateGraph(RuntimeState)
        workflow.add_node("router", self._router_node)
        workflow.add_node("supervisor", self._supervisor_node)
        workflow.add_node("specialists", self._specialists_node)
        workflow.add_node("confirm_action", self._confirm_action_node)
        workflow.add_node("guard", self._guard_node)
        workflow.add_edge(START, "router")
        workflow.add_edge("router", "supervisor")
        workflow.add_edge("supervisor", "specialists")
        workflow.add_conditional_edges(
            "specialists",
            self._next_after_specialists,
            {"confirm_action": "confirm_action", "guard": "guard"},
        )
        workflow.add_edge("confirm_action", "guard")
        workflow.add_edge("guard", END)
        return workflow.compile(checkpointer=self._checkpointer)

    def _router_node(self, state: RuntimeState) -> dict[str, Any]:
        route = self.router.analyze(state["message"])
        return {"route": route.model_dump()}

    def _supervisor_node(self, state: RuntimeState) -> dict[str, Any]:
        route = RouteAnalysis.model_validate(state["route"])
        decision = self.supervisor.plan(state["message"], route)
        return {"decision": decision.model_dump()}

    def _specialists_node(self, state: RuntimeState) -> dict[str, Any]:
        execution = self.specialists.execute(
            message=state["message"],
            customer_id=state["customer_id"],
            route=RouteAnalysis.model_validate(state["route"]),
            decision=SupervisorDecision.model_validate(state["decision"]),
        )
        return {
            "agents_invoked": execution.agents_invoked,
            "specialist_results": execution.results,
            "business_result": execution.result,
            "pending_confirmation": execution.pending_confirmation,
        }

    @staticmethod
    def _next_after_specialists(state: RuntimeState) -> str:
        if state.get("pending_confirmation") is not None:
            return "confirm_action"
        return "guard"

    def _confirm_action_node(self, state: RuntimeState) -> dict[str, Any]:
        action = state["pending_confirmation"]
        if action is None:
            raise ValueError("Missing pending action for confirmation")
        approval = interrupt(
            {
                "status": "pending_confirmation",
                "pending_confirmation": action,
                "reply": self.guard.render(action),
            }
        )
        if approval.get("approved") is True:
            result = self.executor.submit_confirmed_action(action["action_id"], state["customer_id"])
            return {"business_result": result}
        result = self.executor.cancel_pending_action(action["action_id"], state["customer_id"])
        return {"business_result": result, "reply": "已取消本次申请。"}

    def _guard_node(self, state: RuntimeState) -> dict[str, Any]:
        result = state["business_result"]
        if result is None:
            raise ValueError("Missing business result for response rendering")
        reply = state.get("reply") or self.guard.render(result)
        return {"status": "completed", "reply": reply, "pending_confirmation": None}

    @staticmethod
    def _config(conversation_id: str) -> dict[str, dict[str, str]]:
        return {"configurable": {"thread_id": conversation_id}}

    @staticmethod
    def _public_result(state: dict[str, Any]) -> dict[str, Any]:
        interrupts = state.get("__interrupt__")
        if interrupts:
            return dict(interrupts[0].value)
        return {
            "status": state["status"],
            "reply": state["reply"],
            "result": state.get("business_result"),
            "agents_invoked": state.get("agents_invoked", []),
        }
