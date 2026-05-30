# 运行 RAG 检索质量评估，支持在线和离线评估模式。

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import sys
from typing import Any

from langchain_core.documents import Document


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from smart_cs.agents.knowledge import KnowledgeAgent
from smart_cs.config import Settings
from smart_cs.rag.evaluation import EvaluationCase, average_metrics, score_case
from smart_cs.rag.indexing import load_knowledge_documents
from smart_cs.rag.retrieval import RuleBasedQueryRewriter


class OfflineMarkdownStore:
    """Transparent fallback for evaluating orchestration without a Milvus service."""

    def __init__(self, documents: list[Document]) -> None:
        self.documents = documents

    def similarity_search(self, query: str, *, k: int, expr: str, **_kwargs: Any) -> list[Document]:
        category_match = re.fullmatch(r'category == "([^"]+)"', expr)
        if category_match is None:
            raise ValueError("Offline evaluation received an unsafe metadata expression")
        category = category_match.group(1)
        terms = {query[index : index + 2] for index in range(max(0, len(query) - 1))}
        candidates = [
            document for document in self.documents if document.metadata["category"] == category
        ]
        return sorted(
            candidates,
            key=lambda document: sum(
                term in str(document.metadata.get("window_text", document.page_content))
                for term in terms
            ),
            reverse=True,
        )[:k]


def load_cases() -> list[EvaluationCase]:
    rows = json.loads((ROOT / "data" / "evaluation" / "rag_cases.json").read_text(encoding="utf-8"))
    return [EvaluationCase(**row) for row in rows]


def build_agent(settings: Settings, offline: bool) -> tuple[KnowledgeAgent, str]:
    if offline:
        documents = load_knowledge_documents(ROOT / "data" / "knowledge")
        return KnowledgeAgent(OfflineMarkdownStore(documents), RuleBasedQueryRewriter()), "offline_markdown_baseline"

    from smart_cs.rag.embeddings import LocalSentenceEmbeddings
    from smart_cs.rag.vector_store import connect_hybrid_store

    embeddings = LocalSentenceEmbeddings(settings.embedding_model)
    return (
        KnowledgeAgent(connect_hybrid_store(settings, embeddings), RuleBasedQueryRewriter()),
        "milvus_hybrid",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Evaluate the evidence workflow without claiming Milvus retrieval performance.",
    )
    args = parser.parse_args()
    settings = Settings()
    agent, retrieval_mode = build_agent(settings, args.offline)
    cases = load_cases()
    details = []
    for case in cases:
        answer = agent.answer(case.question)
        metrics = score_case(
            case,
            answer=answer.answer,
            contexts=answer.contexts,
            retrieved_ids=[citation.context_id for citation in answer.citations],
        )
        details.append({"question": case.question, "category": case.category, "metrics": metrics})

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "case_count": len(cases),
        "retrieval_mode": retrieval_mode,
        "model_mode": settings.model_mode,
        "embedding_model": settings.embedding_model if not args.offline else "not_used",
        "metrics": average_metrics([detail["metrics"] for detail in details]),
        "cases": details,
    }
    output_directory = ROOT / "data" / "evaluation"
    json_path = output_directory / "latest_results.json"
    markdown_path = output_directory / "latest_results.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    markdown_path.write_text(render_markdown(report), encoding="utf-8")
    print(f"Wrote {json_path} and {markdown_path} using {retrieval_mode}.")


def render_markdown(report: dict[str, Any]) -> str:
    metrics = report["metrics"]
    markdown = (
        "# RAG Acceptance Results\n\n"
        f"- Generated at: `{report['generated_at']}`\n"
        f"- Retrieval mode: `{report['retrieval_mode']}`\n"
        f"- Model mode: `{report['model_mode']}`\n"
        f"- Embedding model: `{report['embedding_model']}`\n"
        f"- Case count: `{report['case_count']}`\n\n"
        "| Metric | Score |\n"
        "| --- | ---: |\n"
        f"| Faithfulness | {metrics['faithfulness']:.4f} |\n"
        f"| Answer relevancy | {metrics['answer_relevancy']:.4f} |\n"
        f"| Context recall | {metrics['context_recall']:.4f} |\n"
        f"| Context precision | {metrics['context_precision']:.4f} |\n"
    )
    if report["retrieval_mode"] == "offline_markdown_baseline":
        markdown += (
            "\n> `offline_markdown_baseline` verifies the evaluation pipeline only; "
            "run without `--offline` against Milvus before making hybrid retrieval claims.\n"
        )
    return markdown


if __name__ == "__main__":
    main()
