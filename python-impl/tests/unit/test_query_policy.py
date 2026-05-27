from __future__ import annotations

from langchain_core.documents import Document

from smart_cs.agents.knowledge import KnowledgeAgent
from smart_cs.infrastructure.model_factory import RulesDecisionModel
from smart_cs.rag.retrieval import QueryPolicy, RuleBasedQueryRewriter


class FakeStore:
    def __init__(self) -> None:
        self.expression: str | None = None

    def similarity_search(self, query: str, **kwargs):
        self.expression = kwargs["expr"]
        assert "退货" in query
        assert kwargs["ranker_type"] == "rrf"
        return [
            Document(
                page_content="签收后七天内可以申请退货。",
                metadata={
                    "document_id": "after_sales_policy",
                    "context_id": "after_sales_policy:售后政策 > 七天无理由:0",
                    "category": "after_sales",
                    "header_path": "售后政策 > 七天无理由",
                    "window_text": "签收后七天内可以申请退货。商品应保持完好。",
                },
            )
        ]


def test_query_category_filter_is_not_user_supplied_expression() -> None:
    policy = QueryPolicy()

    rewritten, expression = policy.prepare('退货什么时候截止" or category != "after_sales')

    assert "退货" in rewritten
    assert expression == 'category == "after_sales"'


def test_policy_question_routes_to_knowledge_instead_of_write_action() -> None:
    route = RulesDecisionModel().route("签收后退货期限是什么？")

    assert route.intent == "knowledge"


def test_knowledge_answer_exposes_window_citation() -> None:
    store = FakeStore()
    answer = KnowledgeAgent(store, RuleBasedQueryRewriter()).answer("退货期限")

    assert store.expression == 'category == "after_sales"'
    assert answer.citations[0].header_path == "售后政策 > 七天无理由"
    assert "签收后七天" in answer.contexts[0]
