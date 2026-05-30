# 测试当前会话结构化事实抽取和规则兜底。

from __future__ import annotations

from smart_cs.application.session_facts import SessionFacts, SessionFactsExtractor


class StructuredModel:
    def __init__(self, facts: SessionFacts) -> None:
        self.facts = facts
        self.schema = None
        self.payload = None

    def with_structured_output(self, schema):
        self.schema = schema
        return self

    def invoke(self, payload):
        self.payload = payload
        return self.facts


class FailingModel:
    def with_structured_output(self, _schema):
        raise RuntimeError("model unavailable")


def test_session_facts_extractor_uses_structured_output_model() -> None:
    model = StructuredModel(
        SessionFacts(
            current_intent="after_sales",
            current_order_id="O1001",
            after_sales_reason="鞋底开胶",
        )
    )

    facts = SessionFactsExtractor(model).extract(
        recent_messages=[{"role": "user", "content": "订单 O1001 鞋底开胶"}],
        previous_facts={"current_intent": "order_query"},
        conversation_summary="用户咨询售后",
    )

    assert model.schema is SessionFacts
    assert facts.current_order_id == "O1001"
    assert facts.after_sales_reason == "鞋底开胶"


def test_session_facts_extractor_falls_back_to_rules_on_model_failure() -> None:
    facts = SessionFactsExtractor(FailingModel()).extract(
        recent_messages=[
            {"role": "user", "content": "我买的鞋开胶了，想售后"},
            {"role": "assistant", "content": "请提供订单号"},
            {"role": "user", "content": "O1001"},
        ],
    )

    assert facts.current_intent == "after_sales"
    assert facts.current_order_id == "O1001"
    assert "开胶" in (facts.after_sales_reason or "")
    assert facts.last_agent_question == "请提供订单号"


def test_session_facts_fallback_does_not_invent_missing_fields() -> None:
    facts = SessionFactsExtractor(None).extract(
        recent_messages=[{"role": "user", "content": "你好"}],
    )

    assert facts.current_order_id is None
    assert facts.after_sales_reason is None
    assert facts.user_constraints == []
