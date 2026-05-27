from __future__ import annotations

from dataclasses import asdict, dataclass
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
                answer="知识库中没有检索到足够依据，请补充问题信息。",
                contexts=[],
                citations=[],
            )
        return KnowledgeAnswer(
            answer=f"根据知识库：{contexts[0]}",
            contexts=contexts,
            citations=citations,
        )
