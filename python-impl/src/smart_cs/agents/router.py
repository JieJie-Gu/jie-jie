from __future__ import annotations

from typing import Protocol

from smart_cs.agents.state import RouteAnalysis


class RoutingDecisionModel(Protocol):
    def route(self, message: str) -> RouteAnalysis: ...


class RouterAgent:
    """Analyze a customer message without choosing or authorizing tools."""

    def __init__(self, decision_model: RoutingDecisionModel) -> None:
        self.decision_model = decision_model

    def analyze(self, message: str) -> RouteAnalysis:
        return self.decision_model.route(message)
