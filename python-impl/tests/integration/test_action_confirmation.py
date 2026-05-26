from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
import logging
from threading import Barrier, Event, enumerate as enumerate_threads
from time import sleep

import pytest
from sqlalchemy import select

from smart_cs.agents.state import RouteAnalysis, SupervisorDecision
from smart_cs.application.agent_runtime import AgentRuntime
from smart_cs.domain.errors import ConversationBusyError, ConversationLeaseLostError, ToolPermissionError
from smart_cs.domain.enums import ActionStatus
from smart_cs.domain.models import Conversation, PendingAction, utc_now
from smart_cs.infrastructure.database import Database
from smart_cs.infrastructure.model_factory import LangChainDecisionModel, RulesDecisionModel
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


def active_heartbeat_threads():
    return [
        thread
        for thread in enumerate_threads()
        if thread.name.startswith("smart-cs-turn-lease-heartbeat-")
    ]


class BlockingRulesDecisionModel(RulesDecisionModel):
    def __init__(self, entered: Event, proceed: Event) -> None:
        self.entered = entered
        self.proceed = proceed

    def route(self, message: str) -> RouteAnalysis:
        if "鞋底开胶" in message:
            self.entered.set()
            if not self.proceed.wait(timeout=5):
                raise TimeoutError("Blocked route was not released")
        return super().route(message)


class FailingRulesDecisionModel(RulesDecisionModel):
    def route(self, _message: str) -> RouteAnalysis:
        raise RuntimeError("injected graph failure")


def test_rules_runtime_logs_non_evaluation_development_mode(tmp_path, caplog) -> None:
    repository = SqlRepository(Database(f"sqlite:///{tmp_path / 'logging-rules.db'}"))
    repository.create_schema()
    with caplog.at_level(logging.WARNING, logger="smart_cs.application.agent_runtime"):
        runtime = AgentRuntime(
            executor=AuthorizedToolExecutor(repository),
            decision_model=RulesDecisionModel(),
            checkpoint_path=tmp_path / "logging-rules-checkpoints.db",
        )
    runtime.close()

    assert "RulesDecisionModel" in caplog.text
    assert "development" in caplog.text
    assert "non-evaluation" in caplog.text


def test_langchain_runtime_does_not_log_rules_development_marker(tmp_path, caplog) -> None:
    class FakeChatModel:
        def with_structured_output(self, _schema):
            return object()

    repository = SqlRepository(Database(f"sqlite:///{tmp_path / 'logging-llm.db'}"))
    repository.create_schema()
    with caplog.at_level(logging.WARNING, logger="smart_cs.application.agent_runtime"):
        runtime = AgentRuntime(
            executor=AuthorizedToolExecutor(repository),
            decision_model=LangChainDecisionModel(FakeChatModel()),
            checkpoint_path=tmp_path / "logging-llm-checkpoints.db",
        )
    runtime.close()

    assert "RulesDecisionModel" not in caplog.text
    assert "non-evaluation" not in caplog.text


def test_approved_after_sales_confirmation_submits_single_ticket(runtime_and_repo) -> None:
    runtime, repository = runtime_and_repo

    pending = runtime.invoke("conv-1", "C001", "订单 O1001 鞋底开胶，申请退款")

    assert pending["status"] == "pending_confirmation"
    assert pending["pending_confirmation"]["action_type"] == "after_sales"
    assert "订单 O1001 当前状态为 delivered。" in pending["reply"]
    assert pending["reply"].endswith("已为您生成售后申请草稿，请确认后提交。")
    assert repository.list_tickets("C001") == []

    completed = runtime.confirm(
        "conv-1", "C001", pending["pending_confirmation"]["action_id"], approved=True
    )

    assert completed["status"] == "completed"
    assert completed["reply"].startswith("售后申请已受理")
    assert len(repository.list_tickets("C001")) == 1


def test_rejected_after_sales_confirmation_cancels_without_ticket(runtime_and_repo) -> None:
    runtime, repository = runtime_and_repo

    pending = runtime.invoke("conv-2", "C001", "订单 O1001 鞋底开胶，申请退款")
    completed = runtime.confirm(
        "conv-2", "C001", pending["pending_confirmation"]["action_id"], approved=False
    )

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

    completed = runtime.confirm(
        conversation_id, "C001", pending["pending_confirmation"]["action_id"], approved=True
    )
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

    completed = runtime.confirm(
        conversation_id, "C001", original["pending_confirmation"]["action_id"], approved=True
    )
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

    replayed = runtime.confirm(conversation_id, "C001", action_id, approved=True)
    repeated = runtime.confirm(conversation_id, "C001", action_id, approved=True)

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
            runtime.confirm(conversation_id, "C001", pending["pending_confirmation"]["action_id"])
        else:
            runtime.confirm(
                conversation_id, "C001", pending["pending_confirmation"]["action_id"], approved=approved
            )

    assert repository.list_tickets("C001") == []
    assert actions(repository)[0].status == "pending_confirmation"

    completed = runtime.confirm(
        conversation_id, "C001", pending["pending_confirmation"]["action_id"], approved=True
    )
    assert completed["result"]["action_id"] == pending["pending_confirmation"]["action_id"]


def test_delayed_confirmation_retry_targets_original_action_not_new_pending(
    runtime_and_repo,
) -> None:
    runtime, repository = runtime_and_repo
    conversation_id = "conv-delayed-confirmation"

    first = runtime.invoke(conversation_id, "C001", "订单 O1001 鞋底开胶，申请退款")
    first_id = first["pending_confirmation"]["action_id"]
    submitted = runtime.confirm(conversation_id, "C001", first_id, approved=True)
    second = runtime.invoke(conversation_id, "C001", "请转人工处理")
    second_id = second["pending_confirmation"]["action_id"]

    retried = runtime.confirm(conversation_id, "C001", first_id, approved=False)

    assert retried["result"]["action_id"] == first_id
    assert retried["result"]["status"] == ActionStatus.SUBMITTED.value
    assert retried["result"]["ticket_id"] == submitted["result"]["ticket_id"]
    assert runtime.executor.pending_action_for_conversation(conversation_id, "C001")[
        "action_id"
    ] == second_id
    cancelled = runtime.confirm(conversation_id, "C001", second_id, approved=False)
    assert cancelled["result"]["action_id"] == second_id
    assert len(repository.list_tickets("C001")) == 1


def test_confirmation_action_must_belong_to_conversation(runtime_and_repo) -> None:
    runtime, repository = runtime_and_repo
    first = runtime.invoke("conv-action-owner-a", "C001", "请转人工处理")
    second = runtime.invoke("conv-action-owner-b", "C001", "请转人工处理")

    with pytest.raises(ToolPermissionError):
        runtime.confirm(
            "conv-action-owner-b",
            "C001",
            first["pending_confirmation"]["action_id"],
            approved=True,
        )

    assert repository.list_tickets("C001") == []
    completed = runtime.confirm(
        "conv-action-owner-b", "C001", second["pending_confirmation"]["action_id"], approved=True
    )
    assert completed["result"]["action_id"] == second["pending_confirmation"]["action_id"]


def test_competing_specialist_drafts_return_canonical_pending_without_mixed_facts(
    tmp_path,
) -> None:
    barrier = Barrier(2)

    class CoordinatedRepository(SqlRepository):
        def create_pending_action(self, *args, **kwargs):
            barrier.wait(timeout=5)
            action_type = kwargs.get("action_type") or args[1]
            if action_type == "after_sales":
                sleep(0.05)
            return super().create_pending_action(*args, **kwargs)

    repository = CoordinatedRepository(Database(f"sqlite:///{tmp_path / 'concurrent-runtime.db'}"))
    repository.create_schema()
    repository.seed_demo_data()
    runtimes = [
        AgentRuntime(
            executor=AuthorizedToolExecutor(repository),
            decision_model=RulesDecisionModel(),
            checkpoint_path=tmp_path / f"concurrent-checkpoints-{position}.db",
        )
        for position in range(2)
    ]
    try:
        repository.claim_conversation("conv-concurrent-draft", "C001")
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [
                pool.submit(
                    runtimes[0].specialists.execute,
                    message="订单 O1001 鞋底开胶，申请退款",
                    customer_id="C001",
                    route=RouteAnalysis(
                        intent="after_sales", entities={"order_id": "O1001"}, risk="medium"
                    ),
                    decision=SupervisorDecision(
                        agents=["OrderAgent", "AfterSalesAgent"],
                        action="draft_after_sales",
                        requires_confirmation=True,
                    ),
                    conversation_id="conv-concurrent-draft",
                    idempotency_key="after-sales-request",
                ),
                pool.submit(
                    runtimes[1].specialists.execute,
                    message="请转人工处理",
                    customer_id="C001",
                    route=RouteAnalysis(intent="handoff", risk="high"),
                    decision=SupervisorDecision(
                        agents=["HandoffAgent"],
                        action="draft_handoff",
                        requires_confirmation=True,
                    ),
                    conversation_id="conv-concurrent-draft",
                    idempotency_key="handoff-request",
                ),
            ]
            executions = [future.result(timeout=10) for future in futures]

        assert all(
            execution.pending_confirmation["action_type"] == "handoff" for execution in executions
        )
        assert all(
            not any(result.get("order_id") == "O1001" for result in execution.results)
            for execution in executions
        )
        pending = repository.get_pending_action("conv-concurrent-draft", "C001")
        assert pending is not None
        assert len([action for action in actions(repository) if action.status == "pending_confirmation"]) == 1
        completed = runtimes[0].confirm(
            "conv-concurrent-draft", "C001", pending.id, approved=True
        )
        assert completed["result"]["action_id"] == pending.id
    finally:
        for runtime in runtimes:
            runtime.close()


def test_same_runtime_rejects_overlapping_turn_before_checkpoint_state_can_mix(tmp_path) -> None:
    repository = SqlRepository(Database(f"sqlite:///{tmp_path / 'same-runtime.db'}"))
    repository.create_schema()
    repository.seed_demo_data()
    entered = Event()
    proceed = Event()
    runtime = AgentRuntime(
        executor=AuthorizedToolExecutor(repository),
        decision_model=BlockingRulesDecisionModel(entered, proceed),
        checkpoint_path=tmp_path / "same-runtime-checkpoints.db",
    )
    try:
        with ThreadPoolExecutor(max_workers=1) as pool:
            first = pool.submit(
                runtime.invoke,
                "conv-shared-thread",
                "C001",
                "订单 O1001 鞋底开胶，申请退款",
            )
            assert entered.wait(timeout=5)
            try:
                with pytest.raises(ConversationBusyError, match="busy"):
                    runtime.invoke("conv-shared-thread", "C001", "请转人工处理")
            finally:
                proceed.set()
            pending = first.result(timeout=10)

        assert pending["pending_confirmation"]["action_type"] == "after_sales"
        completed = runtime.confirm(
            "conv-shared-thread",
            "C001",
            pending["pending_confirmation"]["action_id"],
            approved=True,
        )
        assert completed["agents_invoked"] == ["OrderAgent", "AfterSalesAgent"]
    finally:
        proceed.set()
        runtime.close()


def test_shared_database_and_checkpoint_reject_overlapping_cross_runtime_turn(tmp_path) -> None:
    repository = SqlRepository(Database(f"sqlite:///{tmp_path / 'shared-runtime.db'}"))
    repository.create_schema()
    repository.seed_demo_data()
    entered = Event()
    proceed = Event()
    checkpoint_path = tmp_path / "shared-runtime-checkpoints.db"
    runtimes = [
        AgentRuntime(
            executor=AuthorizedToolExecutor(repository),
            decision_model=BlockingRulesDecisionModel(entered, proceed),
            checkpoint_path=checkpoint_path,
        ),
        AgentRuntime(
            executor=AuthorizedToolExecutor(repository),
            decision_model=RulesDecisionModel(),
            checkpoint_path=checkpoint_path,
        ),
    ]
    try:
        with ThreadPoolExecutor(max_workers=1) as pool:
            first = pool.submit(
                runtimes[0].invoke,
                "conv-shared-runtime-thread",
                "C001",
                "订单 O1001 鞋底开胶，申请退款",
            )
            assert entered.wait(timeout=5)
            try:
                with pytest.raises(ConversationBusyError, match="busy"):
                    runtimes[1].invoke("conv-shared-runtime-thread", "C001", "请转人工处理")
            finally:
                proceed.set()
            pending = first.result(timeout=10)

        assert pending["pending_confirmation"]["action_type"] == "after_sales"
        completed = runtimes[1].confirm(
            "conv-shared-runtime-thread",
            "C001",
            pending["pending_confirmation"]["action_id"],
            approved=True,
        )
        assert completed["agents_invoked"] == ["OrderAgent", "AfterSalesAgent"]
    finally:
        proceed.set()
        for runtime in runtimes:
            runtime.close()


def test_pending_interrupt_releases_turn_lease(runtime_and_repo) -> None:
    runtime, repository = runtime_and_repo

    pending = runtime.invoke("conv-interrupt-release", "C001", "请转人工处理")
    assert active_heartbeat_threads() == []
    repository.acquire_turn_lease("conv-interrupt-release", "C001", "probe", ttl_seconds=30)
    repository.release_turn_lease("conv-interrupt-release", "C001", "probe")

    completed = runtime.confirm(
        "conv-interrupt-release",
        "C001",
        pending["pending_confirmation"]["action_id"],
        approved=False,
    )
    assert completed["result"]["status"] == ActionStatus.CANCELLED.value


def test_graph_exception_releases_turn_lease(tmp_path) -> None:
    repository = SqlRepository(Database(f"sqlite:///{tmp_path / 'failure-release.db'}"))
    repository.create_schema()
    repository.seed_demo_data()
    runtime = AgentRuntime(
        executor=AuthorizedToolExecutor(repository),
        decision_model=FailingRulesDecisionModel(),
        checkpoint_path=tmp_path / "failure-release-checkpoints.db",
    )
    try:
        with pytest.raises(RuntimeError, match="injected graph failure"):
            runtime.invoke("conv-failure-release", "C001", "查询订单 O1001")

        assert active_heartbeat_threads() == []
        repository.acquire_turn_lease("conv-failure-release", "C001", "probe", ttl_seconds=30)
        repository.release_turn_lease("conv-failure-release", "C001", "probe")
    finally:
        runtime.close()


def test_expired_turn_lease_can_be_recovered(tmp_path) -> None:
    repository = SqlRepository(Database(f"sqlite:///{tmp_path / 'expired-lease.db'}"))
    repository.create_schema()
    repository.seed_demo_data()
    repository.claim_conversation("conv-expired-lease", "C001")
    repository.acquire_turn_lease("conv-expired-lease", "C001", "stale", ttl_seconds=30)
    with repository.database.session() as session:
        conversation = session.get(Conversation, "conv-expired-lease")
        assert conversation is not None
        conversation.turn_lease_expires_at = utc_now() - timedelta(seconds=1)

    repository.acquire_turn_lease("conv-expired-lease", "C001", "replacement", ttl_seconds=30)
    repository.release_turn_lease("conv-expired-lease", "C001", "replacement")


def test_heartbeat_keeps_long_running_shared_thread_turn_exclusive(tmp_path) -> None:
    database_url = f"sqlite:///{tmp_path / 'heartbeat-runtime.db'}"
    first_repository = SqlRepository(Database(database_url))
    first_repository.create_schema()
    first_repository.seed_demo_data()
    second_repository = SqlRepository(Database(database_url))
    entered = Event()
    proceed = Event()
    checkpoint_path = tmp_path / "heartbeat-runtime-checkpoints.db"
    runtimes = [
        AgentRuntime(
            executor=AuthorizedToolExecutor(first_repository),
            decision_model=BlockingRulesDecisionModel(entered, proceed),
            checkpoint_path=checkpoint_path,
            turn_lease_ttl_seconds=0.5,
            turn_lease_renew_interval_seconds=0.1,
        ),
        AgentRuntime(
            executor=AuthorizedToolExecutor(second_repository),
            decision_model=RulesDecisionModel(),
            checkpoint_path=checkpoint_path,
            turn_lease_ttl_seconds=0.5,
            turn_lease_renew_interval_seconds=0.1,
        ),
    ]
    try:
        with ThreadPoolExecutor(max_workers=1) as pool:
            first = pool.submit(
                runtimes[0].invoke,
                "conv-heartbeat-thread",
                "C001",
                "订单 O1001 鞋底开胶，申请退款",
            )
            assert entered.wait(timeout=5)
            sleep(0.8)
            try:
                with pytest.raises(ConversationBusyError, match="busy"):
                    runtimes[1].invoke("conv-heartbeat-thread", "C001", "请转人工处理")
            finally:
                proceed.set()
            pending = first.result(timeout=10)

        completed = runtimes[1].confirm(
            "conv-heartbeat-thread",
            "C001",
            pending["pending_confirmation"]["action_id"],
            approved=True,
        )
        assert completed["agents_invoked"] == ["OrderAgent", "AfterSalesAgent"]
    finally:
        proceed.set()
        for runtime in runtimes:
            runtime.close()
    assert active_heartbeat_threads() == []


def test_turn_that_loses_lease_token_aborts_without_returning_success(tmp_path) -> None:
    repository = SqlRepository(Database(f"sqlite:///{tmp_path / 'lost-lease.db'}"))
    repository.create_schema()
    repository.seed_demo_data()
    entered = Event()
    proceed = Event()
    runtime = AgentRuntime(
        executor=AuthorizedToolExecutor(repository),
        decision_model=BlockingRulesDecisionModel(entered, proceed),
        checkpoint_path=tmp_path / "lost-lease-checkpoints.db",
        turn_lease_ttl_seconds=1,
        turn_lease_renew_interval_seconds=0.05,
    )
    try:
        with ThreadPoolExecutor(max_workers=1) as pool:
            first = pool.submit(
                runtime.invoke,
                "conv-lost-lease",
                "C001",
                "订单 O1001 鞋底开胶，申请退款",
            )
            assert entered.wait(timeout=5)
            with repository.database.session() as session:
                conversation = session.get(Conversation, "conv-lost-lease")
                assert conversation is not None
                conversation.turn_lease_token = "replacement-owner"
                conversation.turn_lease_expires_at = utc_now() + timedelta(seconds=30)
            sleep(0.15)
            proceed.set()
            with pytest.raises(ConversationLeaseLostError, match="lease"):
                first.result(timeout=10)

        assert actions(repository) == []
        with pytest.raises(ConversationBusyError, match="busy"):
            repository.acquire_turn_lease("conv-lost-lease", "C001", "intruder", ttl_seconds=30)
        repository.release_turn_lease("conv-lost-lease", "C001", "replacement-owner")
    finally:
        proceed.set()
        runtime.close()
    assert active_heartbeat_threads() == []


def test_close_waits_for_active_turn_and_returns_without_heartbeat_threads(tmp_path) -> None:
    repository = SqlRepository(Database(f"sqlite:///{tmp_path / 'close-heartbeat.db'}"))
    repository.create_schema()
    repository.seed_demo_data()
    entered = Event()
    proceed = Event()
    runtime = AgentRuntime(
        executor=AuthorizedToolExecutor(repository),
        decision_model=BlockingRulesDecisionModel(entered, proceed),
        checkpoint_path=tmp_path / "close-heartbeat-checkpoints.db",
        turn_lease_ttl_seconds=1,
        turn_lease_renew_interval_seconds=0.05,
    )
    closed = False
    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            first = pool.submit(
                runtime.invoke,
                "conv-close-heartbeat",
                "C001",
                "订单 O1001 鞋底开胶，申请退款",
            )
            assert entered.wait(timeout=5)
            closing = pool.submit(runtime.close)
            try:
                sleep(0.1)
                assert not closing.done()
            finally:
                proceed.set()
            assert first.result(timeout=10)["status"] == "pending_confirmation"
            closing.result(timeout=10)
            closed = True
    finally:
        proceed.set()
        if not closed:
            runtime.close()
    assert active_heartbeat_threads() == []
