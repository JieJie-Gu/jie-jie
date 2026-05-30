# 测试长期记忆注入前的过滤、排序和安全投影。
from __future__ import annotations

from datetime import UTC, datetime, timedelta

from smart_cs.application.memory_selector import MemoryContextSelector, MemorySelectionInput


def test_memory_selector_filters_unapproved_low_confidence_high_risk_and_expired() -> None:
    now = datetime.now(UTC)
    memories = [
        {
            "key": "preference:shoe_size",
            "memory_kind": "semantic",
            "memory_type": "preference",
            "title": "鞋码偏好",
            "description": "用户通常穿42码",
            "confidence": "high",
            "risk_level": "low",
            "review_status": "approved",
            "expires_at": (now + timedelta(days=30)).isoformat(),
        },
        {
            "key": "risk",
            "memory_kind": "semantic",
            "memory_type": "risk_event",
            "title": "风险",
            "description": "高风险",
            "confidence": "high",
            "risk_level": "high",
            "review_status": "approved",
        },
        {
            "key": "expired",
            "memory_kind": "semantic",
            "memory_type": "preference",
            "title": "过期偏好",
            "description": "已过期",
            "confidence": "high",
            "risk_level": "low",
            "review_status": "approved",
            "expires_at": (now - timedelta(days=1)).isoformat(),
        },
        {
            "key": "candidate",
            "memory_kind": "semantic",
            "memory_type": "preference",
            "title": "候选",
            "description": "未审核",
            "confidence": "high",
            "risk_level": "low",
            "review_status": "pending",
        },
    ]

    result = MemoryContextSelector().select(
        MemorySelectionInput(query="鞋码 42", memories=memories, limit=5)
    )

    assert [memory.memory_id for memory in result.memories] == ["preference:shoe_size"]
    projected = result.memories[0].model_dump()
    assert "risk_level" not in projected
    assert "review_status" not in projected
    assert "value" not in projected
    assert "evidence" not in projected


def test_memory_selector_boosts_episodic_memory_for_after_sales_intent() -> None:
    result = MemoryContextSelector().select(
        MemorySelectionInput(
            query="售后 订单",
            intent="after_sales",
            memories=[
                {
                    "key": "preference:color",
                    "memory_kind": "semantic",
                    "memory_type": "preference",
                    "title": "颜色偏好",
                    "description": "用户喜欢黑色",
                    "confidence": "high",
                    "risk_level": "low",
                    "review_status": "approved",
                },
                {
                    "key": "episode:after_sales_event:O1001:A1",
                    "memory_kind": "episodic",
                    "memory_type": "after_sales_event",
                    "title": "订单 O1001 售后",
                    "description": "用户曾提交鞋底开胶售后申请",
                    "confidence": "high",
                    "risk_level": "low",
                    "review_status": "approved",
                },
            ],
        )
    )

    assert result.memories[0].memory_kind == "episodic"
