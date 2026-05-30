# 定义对话证据和证据校验相关领域结构。

from pydantic import BaseModel, Field


class VisualEvidence(BaseModel):
    visible_issue: str = Field(description="Visible issue category or uncertain")
    affected_part: str = Field(description="Visible affected product part")
    summary: str = Field(description="Short factual description of visible evidence")
    confidence: float = Field(ge=0.0, le=1.0)
    needs_clarification: bool
    extracted_text: str | None = None

    @property
    def usable_for_draft(self) -> bool:
        return self.confidence >= 0.8 and not self.needs_clarification
