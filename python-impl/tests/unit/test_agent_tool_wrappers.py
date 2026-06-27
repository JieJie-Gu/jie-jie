# 测试 agent 工具 wrapper 的审计、权限和草稿行为。

from __future__ import annotations

import pytest

from smart_cs.application.policy import PolicyEngine
from smart_cs.domain.errors import InvalidActionState, ToolPermissionError
from smart_cs.infrastructure.database import Database
from smart_cs.infrastructure.repositories import SqlRepository
from smart_cs.tools.agent_tool_wrappers import (
    RuntimeToolContext,
    draft_after_sales_action,
    draft_handoff_action,
    make_request_after_sales_tool,
    make_request_handoff_tool,
    run_knowledge_rag,
)
from smart_cs.tools.executor import AuthorizedToolExecutor
from tests.api.support import StaticKnowledgeAgent


@pytest.fixture
def repo(tmp_path):
    database = Database(f"sqlite:///{tmp_path / 'agent-tools.db'}")
    repository = SqlRepository(database)
    repository.create_schema()
    repository.seed_demo_data()
    return repository


@pytest.fixture
def executor(repo):
    return AuthorizedToolExecutor(repo)


def context(
    executor: AuthorizedToolExecutor,
    visual_evidence: dict | None = None,
) -> RuntimeToolContext:
    executor.claim_conversation("conv-1", "C001")
    return RuntimeToolContext(
        conversation_id="conv-1",
        customer_id="C001",
        request_id="req-1",
        turn_fence=None,
        visual_evidence=visual_evidence,
    )


def test_knowledge_rag_returns_citations_and_audits(executor, repo) -> None:
    ctx = context(executor)

    result = run_knowledge_rag(
        executor,
        StaticKnowledgeAgent(),
        ctx,
        query="售后政策",
        caller_agent="PostSalesAgent",
    )

    assert result["citations"]
    call = repo.list_tool_calls("C001")[-1]
    assert call.tool_name == "knowledge_rag"
    assert call.status == "succeeded"
    assert call.result["citations"]


class FailingKnowledgeService:
    def answer(self, _query: str):
        raise RuntimeError("milvus unavailable")


def test_knowledge_rag_dependency_failure_returns_safe_unavailable_result(executor, repo) -> None:
    ctx = context(executor)

    result = run_knowledge_rag(
        executor,
        FailingKnowledgeService(),
        ctx,
        query="售后政策",
        caller_agent="PostSalesAgent",
    )

    assert result["status"] == "knowledge_unavailable"
    assert result["contexts"] == []
    assert result["citations"] == []
    call = repo.list_tool_calls("C001")[-1]
    assert call.tool_name == "knowledge_rag"
    assert call.status == "rejected"
    assert call.error_type == "RuntimeError"
    assert call.result == result


def test_request_after_sales_drafts_pending_without_ticket(executor, repo) -> None:
    ctx = context(executor)

    result = draft_after_sales_action(
        executor,
        StaticKnowledgeAgent(),
        PolicyEngine(),
        ctx,
        order_id="O1001",
        reason="鞋底开胶",
    )

    assert result["status"] == "pending_confirmation"
    assert result["action_type"] == "after_sales"
    assert result["order_id"] == "O1001"
    assert repo.list_tickets("C001") == []


def test_request_handoff_drafts_pending_without_ticket(executor, repo) -> None:
    ctx = context(executor)

    result = draft_handoff_action(executor, ctx, reason="需要人工处理")

    assert result["status"] == "pending_confirmation"
    assert result["action_type"] == "handoff"
    assert repo.list_tickets("C001") == []


def test_lookup_order_policy_blocks_presales_agent(executor) -> None:
    with pytest.raises(ToolPermissionError):
        executor.invoke(
            "lookup_order",
            {"customer_id": "C001", "order_id": "O1001"},
            caller_agent="PreSalesAgent",
        )


def test_low_confidence_visual_evidence_routes_to_handoff(executor, repo) -> None:
    ctx = context(
        executor,
        {
            "summary": "图片模糊，无法确认鞋底问题",
            "confidence": 0.42,
            "needs_clarification": True,
        },
    )

    result = draft_after_sales_action(
        executor,
        StaticKnowledgeAgent(),
        PolicyEngine(),
        ctx,
        order_id="O1001",
        reason="鞋底开胶",
    )

    assert result["status"] == "pending_confirmation"
    assert result["action_type"] == "handoff"
    assert any(call.tool_name == "draft_handoff" for call in repo.list_tool_calls("C001"))


def test_unusable_visual_evidence_does_not_create_after_sales_draft(executor, repo) -> None:
    ctx = context(
        executor,
        {
            "summary": "图片需要补充证据",
            "confidence": 0.9,
            "needs_clarification": True,
        },
    )

    result = draft_after_sales_action(
        executor,
        StaticKnowledgeAgent(),
        PolicyEngine(),
        ctx,
        order_id="O1001",
        reason="鞋底开胶",
    )

    assert result["action_type"] == "handoff"
    assert repo.list_tickets("C001") == []
    successful_after_sales_drafts = [
        call
        for call in repo.list_tool_calls("C001")
        if call.tool_name == "draft_after_sales" and call.status == "succeeded"
    ]
    assert successful_after_sales_drafts == []


def test_approved_after_sales_tool_only_submits_existing_pending_action(executor, repo) -> None:
    ctx = context(executor)
    draft = draft_after_sales_action(
        executor,
        StaticKnowledgeAgent(),
        PolicyEngine(),
        ctx,
        order_id="O1001",
        reason="鞋底开胶",
    )
    before = len(repo.list_tool_calls("C001"))
    request_after_sales = make_request_after_sales_tool(executor, lambda: ctx)

    result = request_after_sales.invoke({"order_id": "O1001", "reason": "鞋底开胶"})
    new_calls = repo.list_tool_calls("C001")[before:]

    assert result["action_id"] == draft["action_id"]
    assert result["status"] == "submitted"
    assert [call.tool_name for call in new_calls] == ["submit_confirmed_action"]
    assert len(repo.list_tickets("C001")) == 1


def test_after_sales_tool_without_pending_action_refuses_direct_submission(executor) -> None:
    ctx = context(executor)
    request_after_sales = make_request_after_sales_tool(executor, lambda: ctx)

    with pytest.raises(InvalidActionState, match="pending"):
        request_after_sales.invoke({"order_id": "O1001", "reason": "鞋底开胶"})


def test_approved_handoff_tool_only_submits_existing_pending_action(executor, repo) -> None:
    ctx = context(executor)
    draft = draft_handoff_action(executor, ctx, reason="需要人工处理")
    before = len(repo.list_tool_calls("C001"))
    request_handoff = make_request_handoff_tool(executor, lambda: ctx)

    result = request_handoff.invoke({"reason": "需要人工处理"})
    new_calls = repo.list_tool_calls("C001")[before:]

    assert result["action_id"] == draft["action_id"]
    assert result["status"] == "submitted"
    assert [call.tool_name for call in new_calls] == ["submit_confirmed_action"]


class RecordingKnowledgeService(StaticKnowledgeAgent):
    def __init__(self) -> None:
        self.queries: list[str] = []

    def answer(self, query: str):
        self.queries.append(query)
        return super().answer(query)


@pytest.mark.parametrize("reason", ["鞋面脱线", "鞋面损坏"])
def test_quality_after_sales_uses_stable_policy_query(executor, reason) -> None:
    ctx = context(executor)
    knowledge = RecordingKnowledgeService()

    result = draft_after_sales_action(
        executor,
        knowledge,
        PolicyEngine(),
        ctx,
        order_id="O1001",
        reason=reason,
    )

    assert result["status"] == "pending_confirmation"
    assert "质量问题" in knowledge.queries[0]
    assert "图片凭证" in knowledge.queries[0]
    assert reason in knowledge.queries[0]
