from __future__ import annotations

from typing import Protocol

from smart_cs.agents.state import RouteAnalysis, RouterContext


class RoutingDecisionModel(Protocol):
    def route(self, context: RouterContext) -> RouteAnalysis: ...


class RouterAgent:
    """Analyze a customer message without choosing or authorizing tools."""

    def __init__(self, decision_model: RoutingDecisionModel) -> None:
        self.decision_model = decision_model

    def analyze(self, context: RouterContext | str) -> RouteAnalysis:
        if isinstance(context, str):
            context = RouterContext(current_message=context)
        return self.decision_model.route(context)
