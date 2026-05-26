from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
import logging
import sqlite3
from pathlib import Path
from threading import Condition, Event, Lock, Thread, local
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
from smart_cs.domain.errors import ConversationLeaseLostError
from smart_cs.infrastructure.model_factory import RulesDecisionModel
from smart_cs.tools.executor import AuthorizedToolExecutor, TurnFence


LOGGER = logging.getLogger(__name__)


class DecisionModel(RoutingDecisionModel, PlanningDecisionModel, Protocol):
    pass


class _TurnLeaseHeartbeat:
    def __init__(
        self,
        *,
        executor: AuthorizedToolExecutor,
        conversation_id: str,
        customer_id: str,
        token: str,
        ttl_seconds: float,
        renew_interval_seconds: float,
    ) -> None:
        self._executor = executor
        self._conversation_id = conversation_id
        self._customer_id = customer_id
        self._token = token
        self._ttl_seconds = ttl_seconds
        self._renew_interval_seconds = renew_interval_seconds
        self._stopped = Event()
        self._failure_lock = Lock()
        self._failure: Exception | None = None
        self._thread = Thread(
            target=self._run,
            name=f"smart-cs-turn-lease-heartbeat-{token}",
            daemon=True,
        )

    def start(self) -> None:
        self._thread.start()

    def turn_fence(self) -> TurnFence:
        return TurnFence(conversation_id=self._conversation_id, lease_token=self._token)

    def stop(self) -> None:
        self._stopped.set()
        self._thread.join()

    def renew_and_check(self) -> None:
        self.check()
        try:
            self._executor.renew_turn_lease(
                self._conversation_id,
                self._customer_id,
                self._token,
                ttl_seconds=self._ttl_seconds,
            )
        except Exception as error:
            self._record_failure(error)
            self.check()

    def check(self) -> None:
        with self._failure_lock:
            failure = self._failure
        if failure is None:
            return
        if isinstance(failure, ConversationLeaseLostError):
            raise failure
        raise ConversationLeaseLostError("Conversation turn lease heartbeat failed") from failure

    def _run(self) -> None:
        while not self._stopped.wait(self._renew_interval_seconds):
            try:
                self._executor.renew_turn_lease(
                    self._conversation_id,
                    self._customer_id,
                    self._token,
                    ttl_seconds=self._ttl_seconds,
                )
            except Exception as error:
                self._record_failure(error)
                return

    def _record_failure(self, error: Exception) -> None:
        with self._failure_lock:
            if self._failure is None:
                self._failure = error


class AgentRuntime:
    """Run customer service orchestration with durable confirmation pauses."""

    TURN_LEASE_TTL_SECONDS = 300.0

    def __init__(
        self,
        *,
        executor: AuthorizedToolExecutor,
        decision_model: DecisionModel,
        checkpoint_path: str | Path,
        turn_lease_ttl_seconds: float = TURN_LEASE_TTL_SECONDS,
        turn_lease_renew_interval_seconds: float | None = None,
    ) -> None:
        renew_interval_seconds = (
            turn_lease_renew_interval_seconds
            if turn_lease_renew_interval_seconds is not None
            else turn_lease_ttl_seconds / 3
        )
        if turn_lease_ttl_seconds <= 0:
            raise ValueError("Turn lease TTL must be positive")
        if renew_interval_seconds <= 0 or renew_interval_seconds >= turn_lease_ttl_seconds:
            raise ValueError("Turn lease renew interval must be positive and shorter than TTL")
        if isinstance(decision_model, RulesDecisionModel):
            LOGGER.warning(
                "RulesDecisionModel enabled: development non-evaluation mode; "
                "do not use this run for evaluation claims."
            )
        self.executor = executor
        self.router = RouterAgent(decision_model)
        self.supervisor = SupervisorAgent(decision_model)
        self.specialists = SpecialistDispatcher(executor)
        self.guard = ResponseGuard()
        self._turn_lease_ttl_seconds = turn_lease_ttl_seconds
        self._turn_lease_renew_interval_seconds = renew_interval_seconds
        self._active_turn = local()
        self._lifecycle = Condition()
        self._active_turn_count = 0
        self._closing = False
        self._closed = False

        checkpoint_file = Path(checkpoint_path)
        checkpoint_file.parent.mkdir(parents=True, exist_ok=True)
        self._checkpoint_connection = sqlite3.connect(str(checkpoint_file), check_same_thread=False)
        self._checkpointer = SqliteSaver(self._checkpoint_connection)
        self.graph = self._build_graph()

    def invoke(self, conversation_id: str, customer_id: str, message: str) -> dict[str, Any]:
        self.executor.claim_conversation(conversation_id, customer_id)
        with self._turn_lease(conversation_id, customer_id):
            pending_action = self.executor.pending_action_for_conversation(
                conversation_id, customer_id
            )
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

        with self._turn_lease(conversation_id, customer_id):
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
        with self._lifecycle:
            self._closing = True
            while self._active_turn_count:
                self._lifecycle.wait()
            if self._closed:
                return
            self._checkpoint_connection.close()
            self._closed = True

    @contextmanager
    def _turn_lease(self, conversation_id: str, customer_id: str) -> Iterator[None]:
        self._begin_turn()
        token = str(uuid4())
        try:
            self.executor.acquire_turn_lease(
                conversation_id,
                customer_id,
                token,
                ttl_seconds=self._turn_lease_ttl_seconds,
            )
            heartbeat = _TurnLeaseHeartbeat(
                executor=self.executor,
                conversation_id=conversation_id,
                customer_id=customer_id,
                token=token,
                ttl_seconds=self._turn_lease_ttl_seconds,
                renew_interval_seconds=self._turn_lease_renew_interval_seconds,
            )
            previous_heartbeat = getattr(self._active_turn, "heartbeat", None)
            self._active_turn.heartbeat = heartbeat
            started = False
            try:
                heartbeat.start()
                started = True
                yield
            finally:
                if started:
                    heartbeat.stop()
                try:
                    if started:
                        heartbeat.renew_and_check()
                finally:
                    try:
                        self.executor.release_turn_lease(conversation_id, customer_id, token)
                    finally:
                        if previous_heartbeat is None:
                            del self._active_turn.heartbeat
                        else:
                            self._active_turn.heartbeat = previous_heartbeat
        finally:
            self._end_turn()

    def _begin_turn(self) -> None:
        with self._lifecycle:
            if self._closing or self._closed:
                raise RuntimeError("Agent runtime is closed")
            self._active_turn_count += 1

    def _end_turn(self) -> None:
        with self._lifecycle:
            self._active_turn_count -= 1
            if self._active_turn_count == 0:
                self._lifecycle.notify_all()

    def _assert_turn_lease(self) -> None:
        heartbeat = getattr(self._active_turn, "heartbeat", None)
        if heartbeat is not None:
            heartbeat.renew_and_check()

    def _current_turn_fence(self) -> TurnFence | None:
        heartbeat = getattr(self._active_turn, "heartbeat", None)
        if heartbeat is None:
            return None
        return heartbeat.turn_fence()

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
        self._assert_turn_lease()
        route = self.router.analyze(state["message"])
        self._assert_turn_lease()
        return {"route": route.model_dump()}

    def _supervisor_node(self, state: RuntimeState) -> dict[str, Any]:
        self._assert_turn_lease()
        route = RouteAnalysis.model_validate(state["route"])
        decision = self.supervisor.plan(state["message"], route)
        self._assert_turn_lease()
        return {"decision": decision.model_dump()}

    def _specialists_node(self, state: RuntimeState) -> dict[str, Any]:
        self._assert_turn_lease()
        execution = self.specialists.execute(
            message=state["message"],
            customer_id=state["customer_id"],
            route=RouteAnalysis.model_validate(state["route"]),
            decision=SupervisorDecision.model_validate(state["decision"]),
            conversation_id=state["conversation_id"],
            idempotency_key=state.get("request_id"),
            turn_fence=self._current_turn_fence(),
        )
        self._assert_turn_lease()
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
        self._assert_turn_lease()
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
        self._assert_turn_lease()
        if not isinstance(approval, dict) or type(approval.get("approved")) is not bool:
            raise ValueError("Confirmation requires boolean approval")
        if approval["approved"]:
            result = self.executor.submit_confirmed_action(
                action["action_id"],
                state["customer_id"],
                turn_fence=self._current_turn_fence(),
            )
            self._assert_turn_lease()
            return {"business_result": result}
        result = self.executor.cancel_pending_action(
            action["action_id"],
            state["customer_id"],
            turn_fence=self._current_turn_fence(),
        )
        self._assert_turn_lease()
        return {"business_result": result}

    def _guard_node(self, state: RuntimeState) -> dict[str, Any]:
        self._assert_turn_lease()
        results = self._response_results(state)
        guarded_contents = self.guard.render_results(results)
        self._assert_turn_lease()
        return {"guarded_contents": guarded_contents}

    def _synthesize_node(self, state: RuntimeState) -> dict[str, Any]:
        self._assert_turn_lease()
        results = self._response_results(state)
        reply = self.supervisor.synthesize(results, state["guarded_contents"])
        self._assert_turn_lease()
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
            return self.executor.submit_confirmed_action(
                action_id, customer_id, turn_fence=self._current_turn_fence()
            )
        return self.executor.cancel_pending_action(
            action_id, customer_id, turn_fence=self._current_turn_fence()
        )

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
