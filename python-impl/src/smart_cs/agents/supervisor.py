from __future__ import annotations

from typing import Any, Protocol

from smart_cs.agents.state import RouteAnalysis, SupervisorDecision


DECLARED_SPECIALISTS = frozenset(
    {"ProductAgent", "OrderAgent", "KnowledgeAgent", "AfterSalesAgent", "HandoffAgent"}
)
WRITE_ACTIONS = frozenset({"draft_after_sales", "draft_handoff"})
WRITE_AGENT_ACTIONS = {
    "AfterSalesAgent": "draft_after_sales",
    "HandoffAgent": "draft_handoff",
}
ACTION_WRITE_AGENTS = {action: agent for agent, action in WRITE_AGENT_ACTIONS.items()}


class PlanningDecisionModel(Protocol):
    def plan(self, message: str, route: RouteAnalysis) -> SupervisorDecision: ...


def validate_decision(decision: SupervisorDecision) -> SupervisorDecision:
    if not decision.agents:
        raise ValueError("Supervisor agent plan must contain at least one agent")

    undeclared_agents = set(decision.agents) - DECLARED_SPECIALISTS
    if undeclared_agents:
        names = ", ".join(sorted(undeclared_agents))
        raise ValueError(f"Supervisor agent plan contains undeclared agent(s): {names}")

    for position, agent in enumerate(decision.agents):
        required_action = WRITE_AGENT_ACTIONS.get(agent)
        if required_action is None:
            continue
        if decision.action != required_action:
            raise ValueError(f"Write agent {agent} does not match action {decision.action}")
        if position != len(decision.agents) - 1:
            raise ValueError(f"Write agent {agent} must be the final agent in the plan")

    required_agent = ACTION_WRITE_AGENTS.get(decision.action)
    if required_agent is not None and decision.agents[-1] != required_agent:
        raise ValueError(f"Action {decision.action} requires {required_agent} as final agent")

    if decision.action == "draft_after_sales" and decision.agents[-2:] != [
        "OrderAgent",
        "AfterSalesAgent",
    ]:
        raise ValueError("Action draft_after_sales requires OrderAgent before AfterSalesAgent")

    if decision.action in WRITE_ACTIONS and not decision.requires_confirmation:
        return decision.model_copy(update={"requires_confirmation": True})
    return decision


class SupervisorAgent:
    """Plan specialist execution and compose only guarded result content."""

    def __init__(self, decision_model: PlanningDecisionModel) -> None:
        self.decision_model = decision_model

    def plan(self, message: str, route: RouteAnalysis) -> SupervisorDecision:
        return validate_decision(self.decision_model.plan(message, route))

    def synthesize(
        self, specialist_results: list[dict[str, Any]], guarded_contents: list[str]
    ) -> str:
        if not specialist_results or len(specialist_results) != len(guarded_contents):
            raise ValueError("Supervisor synthesis requires one guarded content per result")

        terminal_status = specialist_results[-1].get("status")
        if terminal_status in {"submitted", "cancelled"}:
            return guarded_contents[-1]
        return "".join(guarded_contents)
