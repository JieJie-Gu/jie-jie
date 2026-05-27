from smart_cs.agents.vision import VisionAgent
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
