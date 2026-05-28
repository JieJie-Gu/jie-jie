from __future__ import annotations

from fastapi.testclient import TestClient
import pytest

from smart_cs.config import Settings
from smart_cs.domain.errors import ConversationBusyError
from smart_cs.infrastructure.model_factory import RulesDecisionModel
from smart_cs.main import create_app


@pytest.fixture
def client(tmp_path):
    settings = Settings(
        database_url=f"sqlite:///{tmp_path / 'api.db'}",
        checkpoint_path=tmp_path / "checkpoints.db",
        model_mode="rules",
    )
    with TestClient(create_app(settings)) as test_client:
        yield test_client


def create_conversation(client: TestClient, customer_id: str = "C001") -> dict:
    response = client.post("/api/conversations", json={"customer_id": customer_id})
    assert response.status_code == 201
    return response.json()


def send_message(
    client: TestClient, conversation_id: str, message: str, customer_id: str = "C001"
):
    return client.post(
        f"/api/conversations/{conversation_id}/messages",
        json={"customer_id": customer_id, "content": message},
    )


def after_sales_message() -> str:
    return f"O1001 {RulesDecisionModel._after_sales_keywords[0]}"


def order_message() -> str:
    return "O1001"


def product_message() -> str:
    return RulesDecisionModel._product_keywords[-1]


def test_create_conversation_returns_id_and_claims_owner(client: TestClient) -> None:
    created = create_conversation(client)

    assert created["id"]
    assert created["customer_id"] == "C001"

    response = send_message(client, created["id"], order_message(), customer_id="C002")
    assert response.status_code == 403


def test_after_sales_requires_confirmation_then_submits_ticket(client: TestClient) -> None:
    created = create_conversation(client)

    pending_response = send_message(client, created["id"], after_sales_message())

    assert pending_response.status_code == 200
    pending = pending_response.json()
    assert pending["status"] == "pending_confirmation"
    assert "pending_confirmation" not in pending
    assert pending["pending_action"]["action_type"] == "after_sales"
    assert pending["pending_action"]["action_id"]
    assert client.app.state.repository.list_tickets("C001") == []

    completed_response = client.post(
        f"/api/conversations/{created['id']}/actions/confirm",
        json={
            "customer_id": "C001",
            "action_id": pending["pending_action"]["action_id"],
            "approved": True,
        },
    )

    assert completed_response.status_code == 200
    completed = completed_response.json()
    assert completed["status"] == "completed"
    assert completed["result"]["status"] == "submitted"
    assert completed["result"]["ticket_id"]
    assert len(client.app.state.repository.list_tickets("C001")) == 1


def test_reject_confirmation_cancels_action_without_ticket(client: TestClient) -> None:
    created = create_conversation(client)
    pending = send_message(client, created["id"], after_sales_message()).json()

    completed_response = client.post(
        f"/api/conversations/{created['id']}/actions/confirm",
        json={
            "customer_id": "C001",
            "action_id": pending["pending_action"]["action_id"],
            "approved": False,
        },
    )

    assert completed_response.status_code == 200
    completed = completed_response.json()
    assert completed["status"] == "completed"
    assert completed["result"]["status"] == "cancelled"
    assert client.app.state.repository.list_tickets("C001") == []


def test_wrong_customer_cannot_confirm_or_read_tool_calls(client: TestClient) -> None:
    created = create_conversation(client)
    pending = send_message(client, created["id"], after_sales_message()).json()

    confirm_response = client.post(
        f"/api/conversations/{created['id']}/actions/confirm",
        json={
            "customer_id": "C002",
            "action_id": pending["pending_action"]["action_id"],
            "approved": True,
        },
    )
    tool_calls_response = client.get(
        f"/api/conversations/{created['id']}/tool-calls",
        params={"customer_id": "C002"},
    )
    owner_tool_calls_response = client.get(
        f"/api/conversations/{created['id']}/tool-calls",
        params={"customer_id": "C001"},
    )

    assert confirm_response.status_code == 403
    assert tool_calls_response.status_code == 403
    assert owner_tool_calls_response.status_code == 200
    tool_calls = owner_tool_calls_response.json()["tool_calls"]
    assert tool_calls
    assert {call["customer_id"] for call in tool_calls} <= {"C001"}


def test_runs_endpoint_returns_latest_pending_run_with_related_tool_calls(
    client: TestClient,
) -> None:
    created = create_conversation(client)

    pending = send_message(client, created["id"], after_sales_message()).json()
    response = client.get(
        f"/api/conversations/{created['id']}/runs",
        params={"customer_id": "C001"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["tool_calls"]
    run = body["runs"][0]
    assert run["agents"] == ["OrderAgent", "AfterSalesAgent"]
    assert run["status"] == "pending_confirmation"
    assert run["pending_action_id"] == pending["pending_action"]["action_id"]
    assert run["reply"] == pending["reply"]
    assert any(call["tool_name"] == "draft_after_sales" for call in body["tool_calls"])


def test_confirm_records_completed_run_with_final_reply(client: TestClient) -> None:
    created = create_conversation(client)
    pending = send_message(client, created["id"], after_sales_message()).json()

    completed = client.post(
        f"/api/conversations/{created['id']}/actions/confirm",
        json={
            "customer_id": "C001",
            "action_id": pending["pending_action"]["action_id"],
            "approved": True,
        },
    ).json()
    runs = client.get(
        f"/api/conversations/{created['id']}/runs",
        params={"customer_id": "C001"},
    ).json()["runs"]

    assert len(runs) == 1
    assert runs[0]["status"] == "completed"
    assert runs[0]["pending_action_id"] == pending["pending_action"]["action_id"]
    assert runs[0]["reply"] == completed["reply"]


def test_runs_endpoint_includes_confirmation_tool_call_for_same_action(
    client: TestClient,
) -> None:
    first = create_conversation(client)
    second = create_conversation(client)
    first_pending = send_message(client, first["id"], after_sales_message()).json()
    second_pending = send_message(client, second["id"], after_sales_message()).json()

    client.post(
        f"/api/conversations/{first['id']}/actions/confirm",
        json={
            "customer_id": "C001",
            "action_id": first_pending["pending_action"]["action_id"],
            "approved": True,
        },
    )
    client.post(
        f"/api/conversations/{second['id']}/actions/confirm",
        json={
            "customer_id": "C001",
            "action_id": second_pending["pending_action"]["action_id"],
            "approved": True,
        },
    )
    body = client.get(
        f"/api/conversations/{first['id']}/runs",
        params={"customer_id": "C001"},
    ).json()

    submit_calls = [
        call for call in body["tool_calls"] if call["tool_name"] == "submit_confirmed_action"
    ]
    assert [call["arguments"]["action_id"] for call in submit_calls] == [
        first_pending["pending_action"]["action_id"]
    ]


def test_runs_endpoint_only_returns_conversation_related_tool_calls(client: TestClient) -> None:
    first = create_conversation(client)
    second = create_conversation(client)

    send_message(client, first["id"], after_sales_message())
    send_message(client, second["id"], order_message())
    body = client.get(
        f"/api/conversations/{first['id']}/runs",
        params={"customer_id": "C001"},
    ).json()

    assert body["tool_calls"]
    assert {
        call["arguments"].get("conversation_id") for call in body["tool_calls"]
    } == {first["id"]}


def test_tool_calls_endpoint_only_returns_conversation_related_calls(
    client: TestClient,
) -> None:
    first = create_conversation(client)
    second = create_conversation(client)

    send_message(client, first["id"], after_sales_message())
    send_message(client, second["id"], order_message())
    body = client.get(
        f"/api/conversations/{first['id']}/tool-calls",
        params={"customer_id": "C001"},
    ).json()

    assert body["tool_calls"]
    assert {
        call["arguments"].get("conversation_id") for call in body["tool_calls"]
    } == {first["id"]}


def test_busy_conversation_maps_to_conflict(client: TestClient, monkeypatch) -> None:
    created = create_conversation(client)

    def busy_invoke(*_args, **_kwargs):
        raise ConversationBusyError("Conversation is busy with another active turn")

    monkeypatch.setattr(client.app.state.service.runtime, "invoke", busy_invoke)

    response = send_message(client, created["id"], order_message())

    assert response.status_code == 409
    assert "busy" in response.json()["detail"].lower()


def test_product_and_order_read_flows_return_completed(client: TestClient) -> None:
    product_conversation = create_conversation(client)
    product_response = send_message(client, product_conversation["id"], product_message())
    order_conversation = create_conversation(client)
    order_response = send_message(client, order_conversation["id"], order_message())

    assert product_response.status_code == 200
    product = product_response.json()
    assert product["status"] == "completed"
    assert product["agents_invoked"] == ["ProductAgent"]
    assert product["result"]["products"][0]["name"]

    assert order_response.status_code == 200
    order = order_response.json()
    assert order["status"] == "completed"
    assert order["agents_invoked"] == ["OrderAgent"]
    assert order["result"]["order_id"] == "O1001"
