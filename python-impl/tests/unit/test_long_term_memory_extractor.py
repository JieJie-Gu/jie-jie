# 测试长期记忆优先使用 LLM structured output，并携带完整运行时上下文。
from __future__ import annotations

import json

from langchain_core.messages import HumanMessage, SystemMessage

from smart_cs.application.long_term_memory import (
    LongTermMemoryExtraction,
    LongTermMemoryExtractor,
)
from smart_cs.application.memory import MemoryCandidate


class StructuredMemoryModel:
    def __init__(self, result: LongTermMemoryExtraction) -> None:
        self.result = result
        self.schema = None
        self.payload = None

    def with_structured_output(self, schema):
        self.schema = schema
        return self

    def invoke(self, payload):
        self.payload = payload
        return self.result


class FailingMemoryModel:
    def with_structured_output(self, _schema):
        raise RuntimeError("memory model unavailable")


def test_long_term_memory_extractor_uses_llm_structured_output_with_runtime_context() -> None:
    fact = MemoryCandidate(
        scope="customer",
        owner_id="C001",
        memory_kind="semantic",
        memory_type="preference",
        key="preference:shoe_size",
        title="Shoe size preference",
        description="User usually wears size 42.",
        value={"shoe_size": "42"},
        evidence=[{"text": "我一般穿42码"}],
        source="llm_extraction",
        confidence="high",
        risk_level="low",
        review_status="approved",
    )
    model = StructuredMemoryModel(LongTermMemoryExtraction(facts=[fact]))

    result = LongTermMemoryExtractor(model=model).extract(
        {
            "conversation_id": "conv-1",
            "customer_id": "C001",
            "message": "我一般穿42码",
            "recent_messages": [{"role": "user", "content": "我一般穿42码"}],
            "session_facts": {"current_intent": "pre_sales"},
            "conversation_summary": "用户在咨询鞋子。",
            "business_result": {"status": "completed"},
            "pending_confirmation": None,
            "visual_evidence": {"summary": "鞋底开胶"},
        }
    )

    assert model.schema is LongTermMemoryExtraction
    assert isinstance(model.payload[0], SystemMessage)
    assert isinstance(model.payload[1], HumanMessage)
    payload = json.loads(model.payload[1].content)
    assert payload["recent_messages"][0]["content"] == "我一般穿42码"
    assert payload["session_facts"]["current_intent"] == "pre_sales"
    assert payload["conversation_summary"] == "用户在咨询鞋子。"
    assert payload["visual_evidence"]["summary"] == "鞋底开胶"
    assert result[0]["source"] == "llm_extraction"
    assert result[0]["memory_kind"] == "semantic"


def test_long_term_memory_extractor_falls_back_only_when_llm_fails() -> None:
    result = LongTermMemoryExtractor(model=FailingMemoryModel()).extract(
        {
            "conversation_id": "conv-1",
            "customer_id": "C001",
            "message": "我一般穿42码",
            "business_result": {},
        }
    )

    assert result
    assert result[0]["key"] == "preference:shoe_size"
    assert result[0]["source"] == "user_message"
