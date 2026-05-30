# 测试售后业务 PolicyEngine 的确定性决策。

from smart_cs.application.policy import PolicyEngine


def test_after_sales_policy_requires_order_id() -> None:
    decision = PolicyEngine().evaluate_after_sales(
        order_result={"status": "information_required"},
        knowledge_result={"citations": [{"document_id": "after_sales_policy"}]},
    )

    assert decision.next_action == "explain"
    assert decision.reason_code == "ORDER_REQUIRED"
    assert "订单编号" in decision.explanation


def test_after_sales_policy_requires_knowledge_citation() -> None:
    decision = PolicyEngine().evaluate_after_sales(
        order_result={"order_id": "O1001", "status": "delivered"},
        knowledge_result={"citations": []},
    )

    assert decision.next_action == "explain"
    assert decision.reason_code == "POLICY_EVIDENCE_REQUIRED"


def test_after_sales_policy_allows_delivered_order_with_citation() -> None:
    decision = PolicyEngine().evaluate_after_sales(
        order_result={"order_id": "O1001", "status": "delivered"},
        knowledge_result={"citations": [{"document_id": "after_sales_policy"}]},
    )

    assert decision.next_action == "allow_draft"
    assert decision.eligible is True
