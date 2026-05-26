import pytest
from sqlalchemy import select

from smart_cs.application.agent_runtime import AgentRuntime
from smart_cs.domain.errors import ToolPermissionError
from smart_cs.domain.models import PendingAction
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


def actions(repository) -> list[PendingAction]:
    with repository.database.session() as session:
        return list(session.scalars(select(PendingAction).order_by(PendingAction.created_at)))


def test_approved_after_sales_confirmation_submits_single_ticket(runtime_and_repo) -> None:
    runtime, repository = runtime_and_repo

    pending = runtime.invoke("conv-1", "C001", "订单 O1001 鞋底开胶，申请退款")

    assert pending["status"] == "pending_confirmation"
    assert pending["pending_confirmation"]["action_type"] == "after_sales"
    assert "订单 O1001 当前状态为 delivered。" in pending["reply"]
    assert pending["reply"].endswith("已为您生成售后申请草稿，请确认后提交。")
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


def test_order_read_flow_returns_guarded_supervisor_reply(runtime_and_repo) -> None:
    runtime, _repository = runtime_and_repo

    completed = runtime.invoke("conv-read", "C001", "查询订单 O1001")

    assert completed["status"] == "completed"
    assert completed["agents_invoked"] == ["OrderAgent"]
    assert completed["reply"] == "订单 O1001 当前状态为 delivered。"


def test_conversation_owner_rejects_another_customer_without_losing_pending_action(
    runtime_and_repo,
) -> None:
    runtime, repository = runtime_and_repo
    conversation_id = "conv-owned"

    pending = runtime.invoke(conversation_id, "C001", "订单 O1001 鞋底开胶，申请退款")

    with pytest.raises(ToolPermissionError):
        runtime.invoke(conversation_id, "C002", "需要人工协助")

    completed = runtime.confirm(conversation_id, "C001", approved=True)
    assert completed["result"]["action_id"] == pending["pending_confirmation"]["action_id"]
    assert len(actions(repository)) == 1
    assert len(repository.list_tickets("C001")) == 1


def test_new_message_while_pending_reuses_original_draft_and_confirmation_path(
    runtime_and_repo,
) -> None:
    runtime, repository = runtime_and_repo
    conversation_id = "conv-pending"

    original = runtime.invoke(conversation_id, "C001", "订单 O1001 鞋底开胶，申请退款")
    repeated = runtime.invoke(conversation_id, "C001", "请改为转人工处理")

    assert repeated["status"] == "pending_confirmation"
    assert repeated["pending_confirmation"]["action_id"] == original["pending_confirmation"]["action_id"]
    assert repeated["reply"] == original["reply"]
    assert len(actions(repository)) == 1

    completed = runtime.confirm(conversation_id, "C001", approved=True)
    assert completed["result"]["action_id"] == original["pending_confirmation"]["action_id"]


def test_replayed_specialist_node_reuses_draft_for_same_request(runtime_and_repo) -> None:
    runtime, repository = runtime_and_repo
    conversation_id = "conv-draft-replay"

    pending = runtime.invoke(conversation_id, "C001", "订单 O1001 鞋底开胶，申请退款")
    interrupted_state = runtime.graph.get_state(
        {"configurable": {"thread_id": conversation_id}}
    ).values
    replayed = runtime._specialists_node(interrupted_state)

    assert replayed["pending_confirmation"]["action_id"] == pending["pending_confirmation"]["action_id"]
    assert len(actions(repository)) == 1


def test_submitted_action_is_projected_consistently_when_confirmation_node_replays(
    runtime_and_repo,
) -> None:
    runtime, repository = runtime_and_repo
    conversation_id = "conv-submit-replay"

    pending = runtime.invoke(conversation_id, "C001", "订单 O1001 鞋底开胶，申请退款")
    action_id = pending["pending_confirmation"]["action_id"]
    committed = runtime.executor.submit_confirmed_action(action_id, "C001")

    replayed = runtime.confirm(conversation_id, "C001", approved=True)
    repeated = runtime.confirm(conversation_id, "C001", approved=True)

    assert replayed["status"] == "completed"
    assert replayed["result"]["ticket_id"] == committed["ticket_id"]
    assert repeated["result"]["ticket_id"] == committed["ticket_id"]
    assert len(repository.list_tickets("C001")) == 1


@pytest.mark.parametrize("approved", ["false", 1, None])
def test_invalid_approval_payload_keeps_action_pending(runtime_and_repo, approved) -> None:
    runtime, repository = runtime_and_repo
    conversation_id = f"conv-invalid-{approved!s}"

    pending = runtime.invoke(conversation_id, "C001", "订单 O1001 鞋底开胶，申请退款")
    with pytest.raises(ValueError, match="boolean approval"):
        if approved is None:
            runtime.confirm(conversation_id, "C001")
        else:
            runtime.confirm(conversation_id, "C001", approved=approved)

    assert repository.list_tickets("C001") == []
    assert actions(repository)[0].status == "pending_confirmation"

    completed = runtime.confirm(conversation_id, "C001", approved=True)
    assert completed["result"]["action_id"] == pending["pending_confirmation"]["action_id"]
