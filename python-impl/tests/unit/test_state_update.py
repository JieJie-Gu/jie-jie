from smart_cs.agents.state import ConversationSlots, RouteAnalysis
from smart_cs.application.state_update import StateUpdater, carry_slots


def test_slot_carry_inherits_active_order_for_follow_up() -> None:
    route = RouteAnalysis(intent="after_sales", entities={}, risk="medium", turn_type="follow_up")
    slots = ConversationSlots(active_order_id="O1001")

    updated = carry_slots(route, slots)

    assert updated.entities["order_id"] == "O1001"


def test_state_update_records_pending_action_then_clears_after_submit() -> None:
    updater = StateUpdater()
    pending_state = {
        "route": {"intent": "after_sales", "entities": {"order_id": "O1001"}, "risk": "medium"},
        "business_result": {
            "action_id": "A1",
            "action_type": "after_sales",
            "status": "pending_confirmation",
            "order_id": "O1001",
        },
    }

    pending_slots = updater.update(pending_state)["conversation_slots"]
    submitted_slots = updater.update(
        {
            "conversation_slots": pending_slots,
            "route": pending_state["route"],
            "business_result": {
                "action_id": "A1",
                "action_type": "after_sales",
                "status": "submitted",
                "order_id": "O1001",
                "ticket_id": "T1",
            },
        }
    )["conversation_slots"]

    assert pending_slots["pending_action"]["action_id"] == "A1"
    assert pending_slots["action_status"] == "pending_confirmation"
    assert submitted_slots["pending_action"] is None
    assert submitted_slots["active_ticket_id"] == "T1"
    assert submitted_slots["action_status"] == "submitted"
