import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier

import pytest
from sqlalchemy import event, select
from sqlalchemy.exc import IntegrityError

from smart_cs.domain.errors import InvalidActionState, ToolPermissionError
from smart_cs.domain.models import PendingAction
from smart_cs.infrastructure.database import Database
from smart_cs.infrastructure.repositories import SqlRepository
from smart_cs.tools.executor import AuthorizedToolExecutor


SEED_SCRIPT = Path(__file__).parents[2] / "scripts" / "seed_demo_data.py"


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


def test_replayed_draft_with_same_idempotency_key_reuses_pending_action(repo) -> None:
    tools = AuthorizedToolExecutor(repo)
    arguments = {
        "customer_id": "C001",
        "order_id": "O1001",
        "reason": "鞋底开胶",
        "idempotency_key": "request-draft-replay",
    }

    first = tools.invoke("draft_after_sales", arguments)
    repeated = tools.invoke("draft_after_sales", arguments)

    with repo.database.session() as session:
        drafts = list(session.scalars(select(PendingAction)))
    assert repeated["action_id"] == first["action_id"]
    assert len(drafts) == 1


def test_idempotency_key_does_not_expose_another_customers_draft(repo) -> None:
    tools = AuthorizedToolExecutor(repo)
    tools.invoke(
        "draft_handoff",
        {"customer_id": "C001", "reason": "需要人工沟通", "idempotency_key": "private-request"},
    )

    with pytest.raises(ToolPermissionError):
        tools.invoke(
            "draft_handoff",
            {"customer_id": "C002", "reason": "其他请求", "idempotency_key": "private-request"},
        )


def test_conversation_scoped_draft_requires_bound_owner(repo) -> None:
    tools = AuthorizedToolExecutor(repo)
    tools.claim_conversation("owned-conversation", "C001")

    with pytest.raises(ToolPermissionError):
        tools.invoke(
            "draft_handoff",
            {
                "customer_id": "C002",
                "reason": "越权请求",
                "conversation_id": "owned-conversation",
                "idempotency_key": "wrong-owner-request",
            },
        )


def test_competing_conversation_claim_allows_only_one_owner(repo) -> None:
    claim_barrier = Barrier(2)

    def claim(customer_id):
        claim_barrier.wait(timeout=5)
        repo.claim_conversation("claimed-once", customer_id)
        return customer_id

    winners = []
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(claim, customer_id) for customer_id in ("C001", "C002")]
        for future in futures:
            try:
                winners.append(future.result(timeout=10))
            except ToolPermissionError:
                pass

    assert len(winners) == 1
    loser = "C002" if winners[0] == "C001" else "C001"
    with pytest.raises(ToolPermissionError):
        repo.claim_conversation("claimed-once", loser)


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


def test_successful_write_rolls_back_when_success_audit_cannot_be_saved(tmp_path) -> None:
    class InjectedAuditFailure(RuntimeError):
        pass

    class FailingSuccessfulAuditRepository(SqlRepository):
        def record_tool_call(self, *args, **kwargs):
            if kwargs["status"] == "succeeded":
                raise InjectedAuditFailure("injected successful audit failure")
            return super().record_tool_call(*args, **kwargs)

    database = Database(f"sqlite:///{tmp_path / 'atomic-audit.db'}")
    repository = FailingSuccessfulAuditRepository(database)
    repository.create_schema()
    repository.seed_demo_data()

    with pytest.raises(InjectedAuditFailure):
        AuthorizedToolExecutor(repository).invoke(
            "draft_handoff", {"customer_id": "C001", "reason": "需要人工沟通"}
        )

    with database.session() as session:
        assert list(session.scalars(select(PendingAction))) == []
    assert repository.list_tickets("C001") == []
    assert repository.list_tool_calls()[-1].status == "rejected"


def test_competing_submit_and_cancel_cannot_create_conflicting_terminal_state(repo) -> None:
    tools = AuthorizedToolExecutor(repo)
    draft = tools.invoke("draft_handoff", {"customer_id": "C001", "reason": "并发终态校验"})
    transition_barrier = Barrier(2)

    def synchronize_updates(_conn, _cursor, statement, _parameters, _context, _executemany):
        if statement.lstrip().upper().startswith("UPDATE PENDING_ACTIONS"):
            transition_barrier.wait(timeout=5)

    event.listen(repo.database.engine, "before_cursor_execute", synchronize_updates)
    results = []
    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [
                pool.submit(tools.submit_confirmed_action, draft["action_id"], "C001"),
                pool.submit(tools.cancel_pending_action, draft["action_id"], "C001"),
            ]
            for future in futures:
                try:
                    results.append(future.result(timeout=10))
                except InvalidActionState:
                    pass
    finally:
        event.remove(repo.database.engine, "before_cursor_execute", synchronize_updates)

    with repo.database.session() as session:
        action = session.get(PendingAction, draft["action_id"])
    tickets = repo.list_tickets("C001")
    statuses = {result["status"] for result in results}
    assert statuses != {"submitted", "cancelled"}
    if action.status == "submitted":
        assert len(tickets) == 1
    else:
        assert action.status == "cancelled"
        assert tickets == []


@pytest.mark.parametrize(
    ("customer_id", "order_id"),
    [("missing-customer", None), ("C001", "missing-order")],
)
def test_sqlite_enforces_pending_action_foreign_keys(repo, customer_id, order_id) -> None:
    with pytest.raises(IntegrityError):
        repo.create_pending_action(
            customer_id=customer_id,
            action_type="handoff",
            reason="invalid reference",
            order_id=order_id,
        )


def test_search_products_returns_customer_visible_product(repo) -> None:
    tools = AuthorizedToolExecutor(repo)

    result = tools.invoke("search_products", {"query": "跑鞋"})

    assert result["products"][0]["name"] == "轻量跑鞋"


def test_seed_script_creates_default_sqlite_parent_directory(tmp_path) -> None:
    clean_cwd = tmp_path / "clean-project"
    clean_cwd.mkdir()
    environment = os.environ.copy()
    environment.pop("SMART_CS_DATABASE_URL", None)

    completed = subprocess.run(
        [sys.executable, str(SEED_SCRIPT)],
        cwd=clean_cwd,
        env=environment,
        capture_output=True,
        text=True,
        check=True,
    )

    assert (clean_cwd / "data" / "smart_cs.db").exists()
    assert "C001" in completed.stdout
    assert "O1001" in completed.stdout
