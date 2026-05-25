import pytest

from smart_cs.domain.errors import InvalidActionState, ToolPermissionError
from smart_cs.infrastructure.database import Database
from smart_cs.infrastructure.repositories import SqlRepository
from smart_cs.tools.executor import AuthorizedToolExecutor


@pytest.fixture
def repo(tmp_path):
    database = Database(f"sqlite:///{tmp_path / 'tools.db'}")
    repository = SqlRepository(database)
    repository.create_schema()
    repository.seed_demo_data()
    return repository


def test_order_lookup_rejects_another_customer_and_audits_rejection(repo) -> None:
    tools = AuthorizedToolExecutor(repo)

    with pytest.raises(ToolPermissionError):
        tools.invoke("lookup_order", {"customer_id": "C002", "order_id": "O1001"})

    calls = repo.list_tool_calls()
    assert calls[-1].tool_name == "lookup_order"
    assert calls[-1].status == "rejected"


def test_after_sales_only_creates_draft_before_confirmation(repo) -> None:
    tools = AuthorizedToolExecutor(repo)

    result = tools.invoke(
        "draft_after_sales",
        {"customer_id": "C001", "order_id": "O1001", "reason": "鞋底开胶"},
    )

    assert result["status"] == "pending_confirmation"
    assert result["action_type"] == "after_sales"
    assert repo.list_tickets("C001") == []


def test_handoff_is_a_draft_until_confirmed(repo) -> None:
    tools = AuthorizedToolExecutor(repo)

    result = tools.invoke("draft_handoff", {"customer_id": "C001", "reason": "需要人工沟通"})

    assert result["status"] == "pending_confirmation"
    assert result["action_type"] == "handoff"
    assert repo.list_tickets("C001") == []


def test_submit_confirmed_action_is_idempotent(repo) -> None:
    tools = AuthorizedToolExecutor(repo)
    draft = tools.invoke(
        "draft_after_sales",
        {"customer_id": "C001", "order_id": "O1001", "reason": "鞋底开胶"},
    )

    submitted = tools.submit_confirmed_action(draft["action_id"], "C001")
    repeated = tools.submit_confirmed_action(draft["action_id"], "C001")

    assert submitted["status"] == "submitted"
    assert submitted["ticket_id"] == repeated["ticket_id"]
    assert len(repo.list_tickets("C001")) == 1


def test_cancelled_action_cannot_be_submitted(repo) -> None:
    tools = AuthorizedToolExecutor(repo)
    draft = tools.invoke("draft_handoff", {"customer_id": "C001", "reason": "不再需要"})

    cancelled = tools.cancel_pending_action(draft["action_id"], "C001")

    assert cancelled["status"] == "cancelled"
    assert repo.list_tickets("C001") == []
    with pytest.raises(InvalidActionState):
        tools.submit_confirmed_action(draft["action_id"], "C001")


def test_another_customer_cannot_submit_a_pending_action(repo) -> None:
    tools = AuthorizedToolExecutor(repo)
    draft = tools.invoke("draft_handoff", {"customer_id": "C001", "reason": "需要人工沟通"})

    with pytest.raises(ToolPermissionError):
        tools.submit_confirmed_action(draft["action_id"], "C002")

    assert repo.list_tickets("C001") == []
    assert repo.list_tool_calls()[-1].status == "rejected"


def test_cancel_is_idempotent_and_submitted_action_cannot_be_cancelled(repo) -> None:
    tools = AuthorizedToolExecutor(repo)
    cancelled_draft = tools.invoke("draft_handoff", {"customer_id": "C001", "reason": "不再需要"})

    first_cancel = tools.cancel_pending_action(cancelled_draft["action_id"], "C001")
    repeated_cancel = tools.cancel_pending_action(cancelled_draft["action_id"], "C001")

    assert first_cancel["status"] == "cancelled"
    assert repeated_cancel["status"] == "cancelled"

    submitted_draft = tools.invoke("draft_handoff", {"customer_id": "C001", "reason": "转接人工"})
    tools.submit_confirmed_action(submitted_draft["action_id"], "C001")
    with pytest.raises(InvalidActionState):
        tools.cancel_pending_action(submitted_draft["action_id"], "C001")


def test_search_products_returns_customer_visible_product(repo) -> None:
    tools = AuthorizedToolExecutor(repo)

    result = tools.invoke("search_products", {"query": "跑鞋"})

    assert result["products"][0]["name"] == "轻量跑鞋"
