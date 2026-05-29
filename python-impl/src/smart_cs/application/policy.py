from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel


class PolicyDecision(BaseModel):
    eligible: bool
    reason_code: str
    explanation: str
    next_action: Literal["allow_draft", "explain", "handoff"]
    requires_human_review: bool = False


class PolicyEngine:
    def evaluate_after_sales(
        self,
        *,
        order_result: dict[str, Any],
        knowledge_result: dict[str, Any],
        visual_evidence: dict[str, Any] | None = None,
    ) -> PolicyDecision:
        if order_result.get("status") == "information_required" or not order_result.get("order_id"):
            return PolicyDecision(
                eligible=False,
                reason_code="ORDER_REQUIRED",
                explanation="需要先提供订单编号，才能判断售后资格。",
                next_action="explain",
            )
        if not knowledge_result.get("citations"):
            return PolicyDecision(
                eligible=False,
                reason_code="POLICY_EVIDENCE_REQUIRED",
                explanation="需要先检索售后政策依据，才能创建售后草稿。",
                next_action="explain",
            )
        if visual_evidence is not None and visual_evidence.get("usable_for_draft") is False:
            return PolicyDecision(
                eligible=False,
                reason_code="VISUAL_EVIDENCE_UNCERTAIN",
                explanation="图片证据暂不能确认问题，建议转人工审核。",
                next_action="handoff",
                requires_human_review=True,
            )
        if order_result.get("status") in {"delivered", "shipped"}:
            return PolicyDecision(
                eligible=True,
                reason_code="AFTER_SALES_DRAFT_ALLOWED",
                explanation="订单已签收，可创建售后草稿并等待用户确认。",
                next_action="allow_draft",
            )
        return PolicyDecision(
            eligible=False,
            reason_code="ORDER_STATUS_NOT_ELIGIBLE",
            explanation="当前订单状态暂不满足创建售后草稿条件。",
            next_action="explain",
        )
