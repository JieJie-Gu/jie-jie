from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any, Protocol
from uuid import uuid4

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from smart_cs.agents.guardrails import ResponseGuard
from smart_cs.agents.router import RouterAgent, RoutingDecisionModel
from smart_cs.agents.specialists import SpecialistDispatcher
from smart_cs.agents.state import RouteAnalysis, RuntimeState, SupervisorDecision
from smart_cs.agents.supervisor import PlanningDecisionModel, SupervisorAgent
from smart_cs.domain.enums import ActionStatus
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
        self.executor.claim_conversation(conversation_id, customer_id)
        pending_action = self.executor.pending_action_for_conversation(conversation_id, customer_id)
        if pending_action is not None:
            state = self.graph.get_state(self._config(conversation_id)).values
            checkpoint_action = state.get("pending_confirmation") or {}
            if checkpoint_action.get("action_id") != pending_action["action_id"]:
                state = {}
            return self._pending_result(pending_action, state)

        result = self.graph.invoke(
            {
                "conversation_id": conversation_id,
                "customer_id": customer_id,
                "request_id": f"{conversation_id}:{uuid4()}",
                "message": message,
                "route": {},
                "decision": {},
                "agents_invoked": [],
                "specialist_results": [],
                "business_result": None,
                "pending_confirmation": None,
                "guarded_contents": [],
                "reply": None,
                "status": "running",
            },
            config=self._config(conversation_id),
        )
        return self._public_result(result)

    def confirm(
        self,
        conversation_id: str,
        customer_id: str,
        action_id: str,
        *,
        approved: bool | None = None,
    ) -> dict[str, Any]:
        config = self._config(conversation_id)
        self.executor.require_conversation_owner(conversation_id, customer_id)
        if type(approved) is not bool:
            raise ValueError("Confirmation requires boolean approval")

        action = self.executor.action_for_conversation(conversation_id, customer_id, action_id)
        state = self._state_for_action(action_id, self.graph.get_state(config).values)
        if action["status"] != ActionStatus.PENDING_CONFIRMATION.value:
            return self._completed_result(action, state)

        interrupted_action = state.get("pending_confirmation")
        if interrupted_action is None or interrupted_action.get("action_id") != action["action_id"]:
            result = self._transition_action(action["action_id"], customer_id, approved)
            return self._completed_result(result, state)

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
        workflow.add_node("synthesize", self._synthesize_node)
        workflow.add_edge(START, "router")
        workflow.add_edge("router", "supervisor")
        workflow.add_edge("supervisor", "specialists")
        workflow.add_edge("specialists", "guard")
        workflow.add_edge("confirm_action", "guard")
        workflow.add_edge("guard", "synthesize")
        workflow.add_conditional_edges(
            "synthesize",
            self._next_after_synthesis,
            {"confirm_action": "confirm_action", "end": END},
        )
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
            conversation_id=state["conversation_id"],
            idempotency_key=state.get("request_id"),
        )
        return {
            "agents_invoked": execution.agents_invoked,
            "specialist_results": execution.results,
            "business_result": execution.result,
            "pending_confirmation": execution.pending_confirmation,
        }

    @staticmethod
    def _next_after_synthesis(state: RuntimeState) -> str:
        result = state.get("business_result") or {}
        if (
            state.get("pending_confirmation") is not None
            and result.get("status") == ActionStatus.PENDING_CONFIRMATION.value
        ):
            return "confirm_action"
        return "end"

    def _confirm_action_node(self, state: RuntimeState) -> dict[str, Any]:
        action = state["pending_confirmation"]
        if action is None:
            raise ValueError("Missing pending action for confirmation")
        approval = interrupt(
            {
                "status": "pending_confirmation",
                "pending_confirmation": action,
                "reply": state.get("reply") or self.guard.render(action),
            }
        )
        if not isinstance(approval, dict) or type(approval.get("approved")) is not bool:
            raise ValueError("Confirmation requires boolean approval")
        if approval["approved"]:
            result = self.executor.submit_confirmed_action(action["action_id"], state["customer_id"])
            return {"business_result": result}
        result = self.executor.cancel_pending_action(action["action_id"], state["customer_id"])
        return {"business_result": result}

    def _guard_node(self, state: RuntimeState) -> dict[str, Any]:
        results = self._response_results(state)
        return {"guarded_contents": self.guard.render_results(results)}

    def _synthesize_node(self, state: RuntimeState) -> dict[str, Any]:
        results = self._response_results(state)
        reply = self.supervisor.synthesize(results, state["guarded_contents"])
        if self._next_after_synthesis(state) == "confirm_action":
            return {"status": ActionStatus.PENDING_CONFIRMATION.value, "reply": reply}
        return {"status": "completed", "reply": reply, "pending_confirmation": None}

    @staticmethod
    def _config(conversation_id: str) -> dict[str, dict[str, str]]:
        return {"configurable": {"thread_id": conversation_id}}

    def _transition_action(
        self, action_id: str, customer_id: str, approved: bool
    ) -> dict[str, Any]:
        if approved:
            return self.executor.submit_confirmed_action(action_id, customer_id)
        return self.executor.cancel_pending_action(action_id, customer_id)

    def _pending_result(
        self, action: dict[str, Any], state: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        reply = self._synthesize_reply(action, state or {})
        return {
            "status": ActionStatus.PENDING_CONFIRMATION.value,
            "pending_confirmation": action,
            "reply": reply,
        }

    def _completed_result(
        self, result: dict[str, Any], state: dict[str, Any]
    ) -> dict[str, Any]:
        return {
            "status": "completed",
            "reply": self._synthesize_reply(result, state),
            "result": result,
            "agents_invoked": state.get("agents_invoked", []),
        }

    def _synthesize_reply(self, result: dict[str, Any], state: dict[str, Any]) -> str:
        response_results = self._response_results(state, result)
        guarded_contents = self.guard.render_results(response_results)
        return self.supervisor.synthesize(response_results, guarded_contents)

    @staticmethod
    def _state_for_action(action_id: str, state: dict[str, Any]) -> dict[str, Any]:
        state_action = state.get("pending_confirmation") or state.get("business_result") or {}
        if state_action.get("action_id") == action_id:
            return state
        return {}

    @staticmethod
    def _response_results(
        state: dict[str, Any], result: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        terminal_result = result if result is not None else state.get("business_result")
        if terminal_result is None:
            raise ValueError("Missing business result for response rendering")
        results = list(state.get("specialist_results", []))
        if results:
            results[-1] = terminal_result
            return results
        return [terminal_result]

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
