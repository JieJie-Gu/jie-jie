import pytest

from smart_cs.agents.router import RouterAgent
from smart_cs.agents.state import RouteAnalysis, SupervisorDecision
from smart_cs.agents.supervisor import SupervisorAgent, validate_decision
from smart_cs.infrastructure.model_factory import RulesDecisionModel


def test_route_analysis_does_not_authorize_tools() -> None:
    route = RouteAnalysis(intent="after_sales", entities={"order_id": "O1001"}, risk="medium")

    assert "authorized_tools" not in RouteAnalysis.model_fields
    assert "authorized_tools" not in route.model_dump()


def test_write_decision_always_requires_confirmation() -> None:
    decision = validate_decision(
        SupervisorDecision(agents=["OrderAgent", "AfterSalesAgent"], action="draft_after_sales")
    )

    assert decision.requires_confirmation is True


@pytest.mark.parametrize(
    "decision",
    [
        SupervisorDecision(agents=[], action="read"),
        SupervisorDecision.model_construct(agents=["UndeclaredAgent"], action="read"),
    ],
)
def test_invalid_or_empty_agent_plan_is_rejected(decision) -> None:
    with pytest.raises(ValueError, match="agent"):
        validate_decision(decision)


@pytest.mark.parametrize(
    "decision",
    [
        SupervisorDecision(agents=["AfterSalesAgent"], action="read"),
        SupervisorDecision(agents=["OrderAgent"], action="draft_after_sales"),
        SupervisorDecision(agents=["AfterSalesAgent", "OrderAgent"], action="draft_after_sales"),
    ],
)
def test_inconsistent_write_plan_is_rejected(decision) -> None:
    with pytest.raises(ValueError, match="action|final"):
        validate_decision(decision)


def test_rules_agents_plan_after_sales_in_business_order() -> None:
    model = RulesDecisionModel()
    message = "订单 O1001 鞋底开胶，申请退款"

    route = RouterAgent(model).analyze(message)
    decision = SupervisorAgent(model).plan(message, route)

    assert route == RouteAnalysis(
        intent="after_sales", entities={"order_id": "O1001"}, risk="medium"
    )
    assert decision.agents == ["OrderAgent", "AfterSalesAgent"]
    assert decision.action == "draft_after_sales"
    assert decision.requires_confirmation is True
