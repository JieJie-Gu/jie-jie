import pytest
from langgraph.types import Command

from smart_cs.application.agent_runtime import AgentRuntime
from smart_cs.infrastructure.database import Database
from smart_cs.infrastructure.model_factory import RulesDecisionModel
from smart_cs.infrastructure.repositories import SqlRepository
from smart_cs.tools.executor import AuthorizedToolExecutor


@pytest.fixture
def runtime_and_repo(tmp_path):
    repository = SqlRepository(Database(f"sqlite:///{tmp_path / 'runtime.db'}"))
    repository.create_schema()
    repository.seed_demo_data()
    runtime = AgentRuntime(
        executor=AuthorizedToolExecutor(repository),
        decision_model=RulesDecisionModel(),
        checkpoint_path=tmp_path / "checkpoints.db",
    )
    try:
        yield runtime, repository
    finally:
        runtime.close()


def test_approved_after_sales_confirmation_submits_single_ticket(runtime_and_repo) -> None:
    runtime, repository = runtime_and_repo

    pending = runtime.invoke("conv-1", "C001", "订单 O1001 鞋底开胶，申请退款")

    assert pending["status"] == "pending_confirmation"
    assert pending["pending_confirmation"]["action_type"] == "after_sales"
    assert pending["reply"] == "已为您生成售后申请草稿，请确认后提交。"
    assert repository.list_tickets("C001") == []

    completed = runtime.confirm("conv-1", "C001", approved=True)

    assert completed["status"] == "completed"
    assert completed["reply"].startswith("售后申请已受理")
    assert len(repository.list_tickets("C001")) == 1


def test_rejected_after_sales_confirmation_cancels_without_ticket(runtime_and_repo) -> None:
    runtime, repository = runtime_and_repo

    runtime.invoke("conv-2", "C001", "订单 O1001 鞋底开胶，申请退款")
    completed = runtime.confirm("conv-2", "C001", approved=False)

    assert completed["status"] == "completed"
    assert completed["reply"] == "已取消本次申请。"
    assert repository.list_tickets("C001") == []


def test_truthy_non_boolean_approval_does_not_submit_action(runtime_and_repo) -> None:
    runtime, repository = runtime_and_repo
    conversation_id = "conv-string-false"

    runtime.invoke(conversation_id, "C001", "订单 O1001 鞋底开胶，申请退款")
    state = runtime.graph.invoke(
        Command(resume={"approved": "false"}),
        config={"configurable": {"thread_id": conversation_id}},
    )

    assert state["business_result"]["status"] != "submitted"
    assert repository.list_tickets("C001") == []
