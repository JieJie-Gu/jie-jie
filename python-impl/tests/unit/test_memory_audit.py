# 测试 memory_candidates 的人工审核和最小审计记录。
from __future__ import annotations

from smart_cs.infrastructure.database import Database
from smart_cs.infrastructure.repositories import SqlRepository


def _repository(tmp_path) -> SqlRepository:
    repository = SqlRepository(Database(f"sqlite:///{tmp_path / 'memory-review.db'}"))
    repository.create_schema()
    repository.seed_demo_data()
    return repository


def test_list_memory_candidates_only_returns_pending(tmp_path) -> None:
    repository = _repository(tmp_path)
    repository.put_memory(
        ("customer", "C001", "memory_candidates"),
        "preference:shoe_size",
        {
            "memory_kind": "semantic",
            "memory_type": "preference",
            "key": "preference:shoe_size",
            "title": "鞋码偏好",
            "description": "用户通常穿42码",
            "value": {"shoe_size": "42"},
            "evidence": [{"text": "我一般穿42码"}],
            "source": "llm_extraction",
            "confidence": "medium",
            "risk_level": "low",
            "review_status": "pending",
        },
        scope="customer",
        owner_id="C001",
        memory_type="preference",
        source="llm_extraction",
        confidence="medium",
        risk_level="low",
        created_by="system",
    )

    candidates = repository.list_memory_candidates(customer_id="C001")

    assert len(candidates) == 1
    assert candidates[0]["key"] == "preference:shoe_size"
    assert candidates[0]["review_status"] == "pending"


def test_approve_memory_candidate_moves_to_active_and_records_audit(tmp_path) -> None:
    repository = _repository(tmp_path)
    repository.put_memory(
        ("customer", "C001", "memory_candidates"),
        "preference:shoe_size",
        {
            "memory_kind": "semantic",
            "memory_type": "preference",
            "key": "preference:shoe_size",
            "title": "鞋码偏好",
            "description": "用户通常穿42码",
            "value": {"shoe_size": "42"},
            "evidence": [{"text": "我一般穿42码"}],
            "source": "llm_extraction",
            "confidence": "medium",
            "risk_level": "low",
            "review_status": "pending",
        },
        scope="customer",
        owner_id="C001",
        memory_type="preference",
        source="llm_extraction",
        confidence="medium",
        risk_level="low",
        created_by="system",
    )

    approved = repository.approve_memory_candidate(
        candidate_key="preference:shoe_size",
        customer_id="C001",
        reviewer_id="reviewer-1",
        edited_value={"shoe_size": "43"},
    )

    active = repository.get_memory(("customer", "C001", "memories"), "preference:shoe_size")
    calls = repository.list_tool_calls("C001")
    assert approved["review_status"] == "approved"
    assert active is not None
    assert active.value_json["value"] == {"shoe_size": "43"}
    assert any(call.tool_name == "memory_review" for call in calls)


def test_reject_memory_candidate_does_not_create_active_memory_and_records_audit(tmp_path) -> None:
    repository = _repository(tmp_path)
    repository.put_memory(
        ("customer", "C001", "memory_candidates"),
        "preference:color",
        {
            "memory_kind": "semantic",
            "memory_type": "preference",
            "key": "preference:color",
            "title": "颜色偏好",
            "description": "证据不足",
            "value": {"color": "black"},
            "evidence": [{"text": "这次要黑色"}],
            "source": "llm_extraction",
            "confidence": "medium",
            "risk_level": "low",
            "review_status": "pending",
        },
        scope="customer",
        owner_id="C001",
        memory_type="preference",
        source="llm_extraction",
        confidence="medium",
        risk_level="low",
        created_by="system",
    )

    rejected = repository.reject_memory_candidate(
        candidate_key="preference:color",
        customer_id="C001",
        reviewer_id="reviewer-1",
        reason="临时需求",
    )

    active = repository.get_memory(("customer", "C001", "memories"), "preference:color")
    calls = repository.list_tool_calls("C001")
    assert rejected["review_status"] == "rejected"
    assert active is None
    assert any(call.tool_name == "memory_review" for call in calls)
