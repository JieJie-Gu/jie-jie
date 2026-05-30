# 提供 API 测试使用的静态知识库 Agent。

from smart_cs.agents.knowledge import Citation, KnowledgeAnswer


class StaticKnowledgeAgent:
    def answer(self, _query: str) -> KnowledgeAnswer:
        return KnowledgeAnswer(
            answer="根据售后政策：签收后七天内可以申请退货。",
            contexts=["签收后七天内可以申请退货。商品应保持完好。"],
            citations=[
                Citation(
                    document_id="after_sales_policy",
                    context_id="after_sales_policy:售后政策 > 七天无理由:0",
                    header_path="售后政策 > 七天无理由",
                )
            ],
        )
