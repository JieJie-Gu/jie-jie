from __future__ import annotations

from typing import Any

from smart_cs.agents.state import ConversationSlots, RouteAnalysis
from smart_cs.domain.enums import ActionStatus


FOLLOW_UP_TURN_TYPES = {"follow_up", "information_update"}


def carry_slots(route: RouteAnalysis, slots: ConversationSlots) -> RouteAnalysis:
    if (
        route.intent in {"order", "after_sales"}
        and "order_id" not in route.entities
        and route.turn_type in FOLLOW_UP_TURN_TYPES
        and slots.active_order_id
    ):
        entities = {**route.entities, "order_id": slots.active_order_id}
        return route.model_copy(update={"entities": entities})
    return route


class StateUpdater:
    def update(self, state: dict[str, Any]) -> dict[str, Any]:
        slots = ConversationSlots.model_validate(state.get("conversation_slots") or {})
        route = RouteAnalysis.model_validate(state.get("route") or {})
        slots.last_intent = route.intent
        slots.last_entities = dict(route.entities)
        if order_id := route.entities.get("order_id"):
            slots.active_order_id = order_id
        if product_id := route.entities.get("product_id"):
            slots.active_product_id = product_id

        for result in state.get("specialist_results") or []:
            self._apply_result(slots, result)
        if state.get("business_result"):
            self._apply_result(slots, state["business_result"])

        return {"conversation_slots": slots.model_dump()}

    @staticmethod
    def _apply_result(slots: ConversationSlots, result: dict[str, Any]) -> None:
        if order_id := result.get("order_id"):
            slots.active_order_id = str(order_id)
        if product_id := result.get("product_id"):
            slots.active_product_id = str(product_id)
        if ticket_id := result.get("ticket_id"):
            slots.active_ticket_id = str(ticket_id)
        status = result.get("status")
        if status == ActionStatus.PENDING_CONFIRMATION.value:
            slots.pending_action = dict(result)
            slots.action_status = ActionStatus.PENDING_CONFIRMATION.value
        elif status in {ActionStatus.SUBMITTED.value, ActionStatus.CANCELLED.value}:
            slots.pending_action = None
            slots.action_status = str(status)
