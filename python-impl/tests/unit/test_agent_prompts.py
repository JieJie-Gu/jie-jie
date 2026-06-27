# 测试子 Agent prompt 对政策取证和售后权威校验链路的约束。
from smart_cs.infrastructure.prompts import PRE_SALES_AGENT_PROMPT, POST_SALES_AGENT_PROMPT


def test_policy_and_faq_answers_require_knowledge_rag() -> None:
    required = "平台政策、FAQ、确认机制、转人工条件必须调用 knowledge_rag"

    assert required in PRE_SALES_AGENT_PROMPT
    assert required in POST_SALES_AGENT_PROMPT


def test_after_sales_request_delegates_authoritative_checks_to_tool() -> None:
    assert "售后申请直接调用 request_after_sales" in POST_SALES_AGENT_PROMPT
    assert "内部完成订单归属、政策证据和 PolicyEngine 权威校验" in POST_SALES_AGENT_PROMPT
    assert "不要在调用 request_after_sales 前重复调用 lookup_order 或 knowledge_rag" in POST_SALES_AGENT_PROMPT
