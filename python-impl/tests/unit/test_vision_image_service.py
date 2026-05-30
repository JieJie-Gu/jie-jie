# 测试图片消息链路中的 vision_evidence 工具审计。

from __future__ import annotations

from smart_cs.application.conversation_service import ConversationService
from smart_cs.domain.evidence import VisualEvidence
from smart_cs.infrastructure.database import Database
from smart_cs.infrastructure.repositories import SqlRepository


class FakeVisionAgent:
    def inspect(self, _image_data_url: str, _user_message: str) -> VisualEvidence:
        return VisualEvidence(
            visible_issue="uncertain",
            affected_part="shoe",
            summary="图片不清晰，无法确认问题部位",
            confidence=0.42,
            needs_clarification=True,
        )


class FakeAssetStorage:
    def save(
        self,
        conversation_id: str,
        _filename: str,
        _content_type: str,
        _content: bytes,
    ) -> str:
        return f"{conversation_id}/evidence.jpg"


class FakeRuntime:
    def invoke(
        self,
        _conversation_id: str,
        _customer_id: str,
        _message: str,
        *,
        visual_evidence=None,
        asset_key=None,
    ) -> dict:
        return {
            "status": "completed",
            "reply": "已收到图片证据。",
            "result": {"visual_evidence": visual_evidence, "asset_key": asset_key},
            "agents_invoked": [],
        }


def test_image_message_records_vision_tool_call(tmp_path) -> None:
    repository = SqlRepository(Database(f"sqlite:///{tmp_path / 'vision-service.db'}"))
    repository.create_schema()
    repository.seed_demo_data()
    repository.claim_conversation("conv-1", "C001")
    service = ConversationService(
        repository=repository,
        runtime=FakeRuntime(),
        vision_agent=FakeVisionAgent(),
        asset_storage=FakeAssetStorage(),
    )

    result = service.send_message_with_image(
        "conv-1",
        "C001",
        "鞋底坏了",
        "shoe.jpg",
        "image/jpeg",
        b"image-bytes",
    )

    vision_call = next(
        call for call in repository.list_tool_calls("C001") if call.tool_name == "vision_evidence"
    )
    assert vision_call.status == "succeeded"
    assert vision_call.arguments == {
        "conversation_id": "conv-1",
        "customer_id": "C001",
        "asset_key": "conv-1/evidence.jpg",
        "content_type": "image/jpeg",
    }
    assert vision_call.result["summary"] == "图片不清晰，无法确认问题部位"
    assert result["visual_evidence"]["confidence"] == 0.42
    assert result["asset_key"] == "conv-1/evidence.jpg"
