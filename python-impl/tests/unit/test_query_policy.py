# 测试 RAG 查询分类、证据过滤和 citation 结构。
from __future__ import annotations

from langchain_core.documents import Document

from smart_cs.agents.knowledge import KnowledgeService
from smart_cs.rag.retrieval import QueryPolicy, RuleBasedQueryRewriter


class FakeStore:
    def __init__(self) -> None:
        self.expression: str | None = None

    def similarity_search(self, query: str, **kwargs):
        self.expression = kwargs["expr"]
        assert "退货" in query
        assert kwargs["ranker_type"] == "rrf"
        assert kwargs["ranker_params"] == {"k": 60}
        return [
            Document(
                page_content="签收后七天内可以申请退货。商品应保持完好并保留必要配件。",
                metadata={
                    "document_id": "after_sales_policy",
                    "context_id": "after_sales_policy:售后政策 > 七天无理由:0",
                    "category": "after_sales",
                    "header_path": "售后政策 > 七天无理由",
                },
            )
        ]


class IrrelevantStore:
    def similarity_search(self, _query: str, **_kwargs):
        return [
            Document(
                page_content="已发货表示包裹已经交给承运方处理。",
                metadata={
                    "document_id": "shipping_policy",
                    "context_id": "shipping_policy:配送说明 > 发货状态:0",
                    "category": "shipping",
                    "header_path": "配送说明 > 发货状态",
                },
            )
        ]


class ReturnPeriodOnlyStore:
    def similarity_search(self, _query: str, **_kwargs):
        return [
            Document(
                page_content="签收后七天内可以申请退货。商品应保持完好。",
                metadata={
                    "document_id": "after_sales_policy",
                    "context_id": "after_sales_policy:售后政策 > 七天无理由:0",
                    "category": "after_sales",
                    "header_path": "售后政策 > 七天无理由",
                },
            )
        ]


def test_query_category_filter_is_not_user_supplied_expression() -> None:
    policy = QueryPolicy()

    rewritten, expression = policy.prepare('退货什么时候截止 or category != "after_sales"')

    assert "退货" in rewritten
    assert expression == 'category == "after_sales"'


def test_query_policy_routes_common_categories() -> None:
    policy = QueryPolicy()

    assert policy.prepare("物流什么时候更新")[1] == 'category == "shipping"'
    assert policy.prepare("跑鞋怎么保养")[1] == 'category == "product"'
    assert policy.prepare("鞋面沾污后怎么清洁")[1] == 'category == "product"'
    assert policy.prepare("什么时候转人工")[1] == 'category == "faq"'


def test_knowledge_answer_exposes_section_citation() -> None:
    store = FakeStore()
    answer = KnowledgeService(store, RuleBasedQueryRewriter()).answer("退货期限")

    assert store.expression == 'category == "after_sales"'
    assert answer.citations[0].header_path == "售后政策 > 七天无理由"
    assert "签收后七天" in answer.contexts[0]


def test_knowledge_answer_clarifies_when_evidence_is_not_relevant() -> None:
    answer = KnowledgeService(IrrelevantStore(), RuleBasedQueryRewriter()).answer("退货期限")

    assert answer.contexts == []
    assert answer.citations == []
    assert "知识库" in answer.as_result()["answer"]


def test_knowledge_answer_clarifies_when_same_category_evidence_lacks_requested_fact() -> None:
    answer = KnowledgeService(ReturnPeriodOnlyStore(), RuleBasedQueryRewriter()).answer(
        "退货运费谁承担"
    )

    assert answer.contexts == []
    assert answer.citations == []
    assert "知识库" in answer.as_result()["answer"]


def test_knowledge_answer_does_not_answer_realtime_order_status_from_policy_docs() -> None:
    answer = KnowledgeService(ReturnPeriodOnlyStore(), RuleBasedQueryRewriter()).answer(
        "退货订单 O1001 当前状态是什么"
    )

    assert answer.contexts == []
    assert answer.citations == []
    assert "知识库" in answer.as_result()["answer"]
