from __future__ import annotations

from dataclasses import asdict, dataclass
import re
from typing import Any, Protocol

from langchain_core.documents import Document

from smart_cs.rag.retrieval import QueryPolicy, QueryRewriter


class KnowledgeStore(Protocol):
    """知识库向量存储需要实现的最小接口。"""

    def similarity_search(self, query: str, **kwargs: Any) -> list[Document]: ...


@dataclass(frozen=True)
class Citation:
    """答案引用来源，用来把回复追溯到具体文档片段。"""

    document_id: str
    context_id: str
    header_path: str


@dataclass(frozen=True)
class KnowledgeAnswer:
    """KnowledgeAgent 的标准返回值：答案、证据上下文和引用信息。"""

    answer: str
    contexts: list[str]
    citations: list[Citation]

    def as_result(self) -> dict[str, Any]:
        """转换成运行时统一使用的字典结果。"""

        return {
            "status": "knowledge_answer",
            "answer": self.answer,
            "contexts": self.contexts,
            "citations": [asdict(citation) for citation in self.citations],
        }


class KnowledgeAgent:
    """从知识库检索规则依据，并生成带引用的可追溯回答。"""

    # 没有足够证据时不编造答案，提示用户补充问题信息。
    INSUFFICIENT_EVIDENCE_MESSAGE = "知识库中没有检索到足够依据，请补充问题信息。"
    # 针对常见事实型问题，要求命中的知识片段必须包含对应证据词，降低误召回。
    FACT_EVIDENCE_TERMS = (
        (("运费", "邮费", "承担"), ("运费", "承担", "规则")),
        (("几天", "多久", "截止", "期限"), ("七天", "天内", "签收", "期限")),
        (("保持", "完好"), ("保持", "完好")),
        (("凭证", "证据", "证明"), ("凭证", "证据", "证明", "照片")),
        (("已发货", "发货状态"), ("已发货", "发货", "包裹", "承运")),
        (("物流更新", "配送状态"), ("物流", "配送", "状态", "更新")),
        (("保养",), ("保养", "清洁", "晾干")),
        (("尺码",), ("尺码", "详情页", "参考")),
        (("人工",), ("人工", "用户明确要求")),
        (("确认", "提交"), ("确认", "提交")),
    )
    # 订单号通常代表实时订单查询，应交给 post_sales_agent 的订单工具，而不是静态知识库。
    REALTIME_ORDER_PATTERN = re.compile(r"\bO\d+\b", re.IGNORECASE)

    def __init__(self, store: KnowledgeStore, rewriter: QueryRewriter | None = None) -> None:
        # store 负责真正的相似度检索；policy 负责查询改写和分类过滤条件。
        self.store = store
        self.policy = QueryPolicy(rewriter)

    def answer(self, query: str) -> KnowledgeAnswer:
        # 先把用户原始问题改写为更适合检索的查询，并生成知识分类过滤表达式。
        rewritten_query, category_expression = self.policy.prepare(query)
        # 在向量库中做混合检索：取前 4 条，使用分类过滤表达式和 RRF 排序。
        documents = self.store.similarity_search(
            rewritten_query,
            k=4,
            expr=category_expression,
            ranker_type="rrf",
            ranker_params={"k": 60},
        )
        # 二次过滤：只保留分类一致、不是实时订单查询、且包含足够证据词的片段。
        documents = [
            document
            for document in documents
            if self._has_relevant_evidence(document, rewritten_query, category_expression)
        ]
        # 优先使用窗口文本作为上下文，缺失时回退到文档正文。
        contexts = [
            str(document.metadata.get("window_text", document.page_content))
            for document in documents
        ]
        # 从 metadata 中提取引用信息，保证答案可以追溯到原始知识片段。
        citations = [
            Citation(
                document_id=str(document.metadata["document_id"]),
                context_id=str(document.metadata["context_id"]),
                header_path=str(document.metadata["header_path"]),
            )
            for document in documents
        ]
        if not contexts:
            # 没有证据就返回固定兜底话术，不让模型/系统凭空回答。
            return KnowledgeAnswer(
                answer=self.INSUFFICIENT_EVIDENCE_MESSAGE,
                contexts=[],
                citations=[],
            )
        # 当前实现直接用最相关的第一段上下文生成答案，并附带全部命中上下文和引用。
        return KnowledgeAnswer(
            answer=f"根据知识库：{contexts[0]}",
            contexts=contexts,
            citations=citations,
        )

    def _has_relevant_evidence(
        self, document: Document, rewritten_query: str, category_expression: str
    ) -> bool:
        # 只接受形如 category == "xxx" 的分类表达式，避免无分类约束的泛召回。
        category_match = re.fullmatch(r'category == "([^"]+)"', category_expression)
        if category_match is None:
            return False
        category = category_match.group(1)
        if document.metadata.get("category") != category:
            return False
        if self._is_realtime_order_query(rewritten_query):
            return False

        evidence_text = str(document.metadata.get("window_text", document.page_content))
        required_terms = self._required_evidence_terms(rewritten_query)
        if required_terms:
            # 对运费、时限、凭证等事实型问题，必须命中专门证据词。
            return any(term in evidence_text for term in required_terms)

        # 其他问题至少需要命中改写查询中的有效二字词。
        return any(term in evidence_text for term in self._query_terms(rewritten_query))

    @classmethod
    def _is_realtime_order_query(cls, query: str) -> bool:
        """识别实时订单状态类问题，避免 KnowledgeAgent 回答动态业务状态。"""

        if cls.REALTIME_ORDER_PATTERN.search(query):
            return True
        return "订单" in query and any(term in query for term in ("当前", "状态", "到哪", "查询"))

    @classmethod
    def _required_evidence_terms(cls, query: str) -> tuple[str, ...]:
        """根据问题关键词返回必须在证据片段中出现的词。"""

        for query_terms, evidence_terms in cls.FACT_EVIDENCE_TERMS:
            if any(term in query for term in query_terms):
                return evidence_terms
        return ()

    def _query_terms(self, query: str) -> set[str]:
        """把查询切成简单二字词，并过滤停用词和分类词。"""

        normalized = re.sub(r"[\s，。！？；：、,.!?;:\"'“”‘’（）()]+", "", query)
        stop_terms = {"什么", "怎么", "如何", "可以", "请问", "是否", "需要"}
        category_terms = {
            term for terms in self.policy.CATEGORY_TERMS.values() for term in terms
        }
        return {
            normalized[index : index + 2]
            for index in range(max(0, len(normalized) - 1))
            if normalized[index : index + 2] not in stop_terms
            and normalized[index : index + 2] not in category_terms
        }
