# 测试 VisionAgent 的结构化视觉证据输出。

from smart_cs.agents.vision import LangChainVisionModel, VisionAgent
from smart_cs.domain.evidence import VisualEvidence


class FakeVisionModel:
    def examine(self, _image_data_url: str, _user_message: str) -> VisualEvidence:
        return VisualEvidence(
            visible_issue="uncertain",
            affected_part="shoe",
            summary="图片不清晰，无法确认问题部位",
            confidence=0.42,
            needs_clarification=True,
        )


def test_low_confidence_image_cannot_support_after_sales_draft() -> None:
    evidence = VisionAgent(FakeVisionModel()).inspect("data:image/jpeg;base64,eA==", "鞋底坏了")

    assert evidence.usable_for_draft is False


class FakeStructuredVisionModel:
    def __init__(self) -> None:
        self.messages = None

    def invoke(self, messages):
        self.messages = messages
        return {
            "visible_issue": "uncertain",
            "affected_part": "shoe",
            "summary": "图片模糊，无法确认鞋底问题",
            "confidence": 0.4,
            "needs_clarification": True,
        }


class FakeChatModel:
    def __init__(self) -> None:
        self.schema = None
        self.structured_model = FakeStructuredVisionModel()

    def with_structured_output(self, schema):
        self.schema = schema
        return self.structured_model


def test_vision_agent_structured_output() -> None:
    chat_model = FakeChatModel()
    vision_model = LangChainVisionModel(chat_model)

    evidence = vision_model.examine("data:image/jpeg;base64,eA==", "鞋底坏了")

    assert chat_model.schema is VisualEvidence
    assert isinstance(evidence, VisualEvidence)
    assert evidence.usable_for_draft is False
    assert chat_model.structured_model.messages is not None
