from __future__ import annotations

from typing import Any, Protocol

from langchain_core.messages import HumanMessage, SystemMessage

from smart_cs.domain.evidence import VisualEvidence


class VisionEvidenceModel(Protocol):
    def examine(self, image_data_url: str, user_message: str) -> VisualEvidence: ...


class LangChainVisionModel:
    """Use structured multimodal output while forbidding action approval."""

    def __init__(self, chat_model: Any) -> None:
        self.model = chat_model.with_structured_output(VisualEvidence)

    def examine(self, image_data_url: str, user_message: str) -> VisualEvidence:
        result = self.model.invoke(
            [
                SystemMessage(
                    content="仅提取图片中可见的售后证据，不判断责任，不批准退款或售后。"
                ),
                HumanMessage(
                    content=[
                        {"type": "text", "text": user_message},
                        {"type": "image_url", "image_url": {"url": image_data_url}},
                    ]
                ),
            ]
        )
        return VisualEvidence.model_validate(result)


class VisionAgent:
    def __init__(self, vision_model: VisionEvidenceModel) -> None:
        self.vision_model = vision_model

    def inspect(self, image_data_url: str, user_message: str) -> VisualEvidence:
        return self.vision_model.examine(image_data_url, user_message)
