from __future__ import annotations

from fastapi.testclient import TestClient

from smart_cs.agents.vision import VisionAgent
from smart_cs.config import Settings
from smart_cs.domain.evidence import VisualEvidence
from smart_cs.infrastructure.assets import LocalAssetStorage
from smart_cs.main import create_app


class ClearDamageModel:
    def examine(self, _image_data_url: str, _message: str) -> VisualEvidence:
        return VisualEvidence(
            visible_issue="sole_separation",
            affected_part="shoe_sole",
            summary="鞋底边缘可见开胶",
            confidence=0.93,
            needs_clarification=False,
        )


class UncertainModel:
    def examine(self, _image_data_url: str, _message: str) -> VisualEvidence:
        return VisualEvidence(
            visible_issue="uncertain",
            affected_part="unknown",
            summary="图片证据暂不能确认问题",
            confidence=0.3,
            needs_clarification=True,
        )


def test_image_message_returns_pending_after_sales_and_evidence_summary(tmp_path) -> None:
    client = make_client(tmp_path, ClearDamageModel())
    with client:
        conversation_id = create_conversation(client)
        response = post_image(client, conversation_id, "订单 O1001 鞋底开胶，申请售后")

    assert response.status_code == 200
    body = response.json()
    assert body["pending_action"]["action_type"] == "after_sales"
    assert body["visual_evidence"]["summary"] == "鞋底边缘可见开胶"
    assert "退款已完成" not in body["reply"]


def test_uncertain_image_creates_confirmable_handoff_instead_of_after_sales(tmp_path) -> None:
    client = make_client(tmp_path, UncertainModel())
    with client:
        conversation_id = create_conversation(client)
        response = post_image(client, conversation_id, "订单 O1001 鞋底开胶，申请退款")

    body = response.json()
    assert body["pending_action"]["action_type"] == "handoff"
    assert body["visual_evidence"]["needs_clarification"] is True
    assert "退款已完成" not in body["reply"]


def make_client(tmp_path, model) -> TestClient:
    settings = Settings(
        database_url=f"sqlite:///{tmp_path / 'api.db'}",
        checkpoint_path=tmp_path / "checkpoints.db",
        model_mode="rules",
    )
    return TestClient(
        create_app(
            settings,
            vision_agent=VisionAgent(model),
            asset_storage=LocalAssetStorage(tmp_path / "assets"),
        )
    )


def create_conversation(client: TestClient) -> str:
    return client.post("/api/conversations", json={"customer_id": "C001"}).json()["id"]


def post_image(client: TestClient, conversation_id: str, content: str):
    return client.post(
        f"/api/conversations/{conversation_id}/messages-with-image",
        data={"customer_id": "C001", "content": content},
        files={"image": ("damage.jpg", b"jpeg-data", "image/jpeg")},
    )
