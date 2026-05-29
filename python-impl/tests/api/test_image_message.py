from __future__ import annotations

from fastapi.testclient import TestClient

from smart_cs.agents.vision import VisionAgent
from smart_cs.config import Settings
from smart_cs.domain.evidence import VisualEvidence
from smart_cs.infrastructure.assets import LocalAssetStorage
from smart_cs.infrastructure.model_factory import RulesDecisionModel
from smart_cs.main import create_app
from tests.api.support import StaticKnowledgeAgent


class ClearDamageModel:
    def examine(self, _image_data_url: str, _message: str) -> VisualEvidence:
        return VisualEvidence(
            visible_issue="sole_separation",
            affected_part="shoe_sole",
            summary="visible sole separation",
            confidence=0.93,
            needs_clarification=False,
        )


class UncertainModel:
    def examine(self, _image_data_url: str, _message: str) -> VisualEvidence:
        return VisualEvidence(
            visible_issue="uncertain",
            affected_part="unknown",
            summary="image evidence is uncertain",
            confidence=0.3,
            needs_clarification=True,
        )


def test_image_message_returns_pending_after_sales_and_evidence_summary(tmp_path) -> None:
    client = make_client(tmp_path, ClearDamageModel())
    with client:
        conversation_id = create_conversation(client)
        response = post_image(client, conversation_id, after_sales_message())

    assert response.status_code == 200
    body = response.json()
    assert body["pending_action"]["action_type"] == "after_sales"
    assert body["visual_evidence"]["summary"] == "visible sole separation"


def test_uncertain_image_creates_confirmable_handoff_instead_of_after_sales(tmp_path) -> None:
    client = make_client(tmp_path, UncertainModel())
    with client:
        conversation_id = create_conversation(client)
        response = post_image(client, conversation_id, after_sales_message())

    body = response.json()
    assert body["pending_action"]["action_type"] == "handoff"
    assert body["visual_evidence"]["needs_clarification"] is True


def test_image_message_run_records_vision_and_workflow_agents_before_ticket(
    tmp_path,
) -> None:
    client = make_client(tmp_path, ClearDamageModel())
    with client:
        conversation_id = create_conversation(client)
        pending = post_image(client, conversation_id, after_sales_message()).json()
        runs_response = client.get(
            f"/api/conversations/{conversation_id}/runs",
            params={"customer_id": "C001"},
        )
        tickets = client.app.state.repository.list_tickets("C001")

    assert runs_response.status_code == 200
    run = runs_response.json()["runs"][0]
    assert run["agents"] == [
        "VisionAgent",
        "OrderAgent",
        "KnowledgeAgent",
        "AfterSalesAgent",
    ]
    assert run["status"] == "pending_confirmation"
    assert run["pending_action_id"] == pending["pending_action"]["action_id"]
    assert tickets == []


def test_image_message_persists_asset_and_visual_evidence_on_message(tmp_path) -> None:
    client = make_client(tmp_path, ClearDamageModel())
    with client:
        conversation_id = create_conversation(client)
        body = post_image(client, conversation_id, after_sales_message()).json()
        message = client.app.state.repository.latest_message(conversation_id)

    assert message is not None
    assert message.asset_key == body["asset_key"]
    assert message.visual_evidence["summary"] == "visible sole separation"


def make_client(tmp_path, model) -> TestClient:
    settings = Settings(
        database_url=f"sqlite:///{tmp_path / 'api.db'}",
        checkpoint_path=tmp_path / "checkpoints.db",
        model_mode="rules",
        rag_enabled=False,
    )
    return TestClient(
        create_app(
            settings,
            knowledge_agent=StaticKnowledgeAgent(),
            vision_agent=VisionAgent(model),
            asset_storage=LocalAssetStorage(tmp_path / "assets"),
        )
    )


def create_conversation(client: TestClient) -> str:
    return client.post("/api/conversations", json={"customer_id": "C001"}).json()["id"]


def after_sales_message() -> str:
    return f"O1001 {RulesDecisionModel._after_sales_keywords[0]}"


def post_image(client: TestClient, conversation_id: str, content: str):
    return client.post(
        f"/api/conversations/{conversation_id}/messages-with-image",
        data={"customer_id": "C001", "content": content},
        files={"image": ("damage.jpg", b"jpeg-data", "image/jpeg")},
    )
