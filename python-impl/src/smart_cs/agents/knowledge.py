from __future__ import annotations

from dataclasses import asdict, dataclass
import re
from typing import Any, Protocol

from langchain_core.documents import Document

from smart_cs.rag.retrieval import QueryPolicy, QueryRewriter


class KnowledgeStore(Protocol):
    def similarity_search(self, query: str, **kwargs: Any) -> list[Document]: ...


@dataclass(frozen=True)
class Citation:
    document_id: str
    context_id: str
    header_path: str


@dataclass(frozen=True)
class KnowledgeAnswer:
    answer: str
    contexts: list[str]
    citations: list[Citation]

    def as_result(self) -> dict[str, Any]:
        return {
            "status": "knowledge_answer",
            "answer": self.answer,
            "contexts": self.contexts,
            "citations": [asdict(citation) for citation in self.citations],
        }


class KnowledgeAgent:
    """Retrieve curated policy evidence and produce an attributable response."""

    INSUFFICIENT_EVIDENCE_MESSAGE = "知识库中没有检索到足够依据，请补充问题信息。"
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
    REALTIME_ORDER_PATTERN = re.compile(r"\bO\d+\b", re.IGNORECASE)

    def __init__(self, store: KnowledgeStore, rewriter: QueryRewriter | None = None) -> None:
        self.store = store
        self.policy = QueryPolicy(rewriter)

    def answer(self, query: str) -> KnowledgeAnswer:
        rewritten_query, category_expression = self.policy.prepare(query)
        documents = self.store.similarity_search(
            rewritten_query,
            k=4,
            expr=category_expression,
            ranker_type="rrf",
            ranker_params={"k": 60},
        )
        documents = [
            document
            for document in documents
            if self._has_relevant_evidence(document, rewritten_query, category_expression)
        ]
        contexts = [
            str(document.metadata.get("window_text", document.page_content))
            for document in documents
        ]
        citations = [
            Citation(
                document_id=str(document.metadata["document_id"]),
                context_id=str(document.metadata["context_id"]),
                header_path=str(document.metadata["header_path"]),
            )
            for document in documents
        ]
        if not contexts:
            return KnowledgeAnswer(
                answer=self.INSUFFICIENT_EVIDENCE_MESSAGE,
                contexts=[],
                citations=[],
            )
        return KnowledgeAnswer(
            answer=f"根据知识库：{contexts[0]}",
            contexts=contexts,
            citations=citations,
        )

    def _has_relevant_evidence(
        self, document: Document, rewritten_query: str, category_expression: str
    ) -> bool:
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
            return any(term in evidence_text for term in required_terms)

        return any(term in evidence_text for term in self._query_terms(rewritten_query))

    @classmethod
    def _is_realtime_order_query(cls, query: str) -> bool:
        if cls.REALTIME_ORDER_PATTERN.search(query):
            return True
        return "订单" in query and any(term in query for term in ("当前", "状态", "到哪", "查询"))

    @classmethod
    def _required_evidence_terms(cls, query: str) -> tuple[str, ...]:
        for query_terms, evidence_terms in cls.FACT_EVIDENCE_TERMS:
            if any(term in query for term in query_terms):
                return evidence_terms
        return ()

    def _query_terms(self, query: str) -> set[str]:
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
