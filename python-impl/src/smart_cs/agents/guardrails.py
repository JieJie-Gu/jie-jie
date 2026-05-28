from __future__ import annotations

from typing import Any


class ResponseGuard:
    """Render only facts returned from deterministic specialist operations."""

    def render_results(self, results: list[dict[str, Any]]) -> list[str]:
        return [self.render(result) for result in results]

    def render(self, result: dict[str, Any]) -> str:
        status = result.get("status")
        action_type = result.get("action_type")
        if status == "pending_confirmation" and action_type == "after_sales":
            return "已为您生成售后申请草稿，请确认后提交。"
        if (
            status == "pending_confirmation"
            and action_type == "handoff"
            and result.get("evidence_status") == "uncertain"
        ):
            return "图片证据暂不能确认问题，已为您生成转人工申请草稿，请确认。"
        if status == "pending_confirmation" and action_type == "handoff":
            return "已为您生成转人工申请草稿，请确认后提交。"
        if status == "submitted" and action_type == "after_sales":
            return f"售后申请已受理，工单编号为 {result['ticket_id']}。"
        if status == "submitted" and action_type == "handoff":
            return f"人工服务申请已提交，工单编号为 {result['ticket_id']}。"
        if status == "cancelled" and action_type in {"after_sales", "handoff"}:
            return "已取消本次申请。"
        if status == "knowledge_answer":
            return str(result["answer"])
        if "message" in result:
            return str(result["message"])
        if "order_id" in result:
            return f"订单 {result['order_id']} 当前状态为 {result['status']}。"
        products = result.get("products")
        if products:
            return f"查询到商品：{products[0]['name']}。"
        if products == []:
            return "未查询到匹配商品。"
        return "已完成处理。"
