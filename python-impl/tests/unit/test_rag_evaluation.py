import importlib.util
from pathlib import Path

from smart_cs.rag.evaluation import EvaluationCase, score_case, score_contexts


def load_evaluate_rag_module():
    script_path = Path(__file__).parents[2] / "scripts" / "evaluate_rag.py"
    spec = importlib.util.spec_from_file_location("evaluate_rag_script", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_context_precision_and_recall_on_labelled_case() -> None:
    case = EvaluationCase(
        question="签收后几天可以申请退货？",
        expected_answer_points=["签收后七天内"],
        expected_context_ids=["policy-returns-1", "policy-condition-1"],
        category="after_sales",
    )

    result = score_contexts(case, retrieved_ids=["policy-returns-1", "shipping-2"])

    assert result["context_precision"] == 0.5
    assert result["context_recall"] == 0.5


def test_case_report_contains_only_the_four_approved_metrics() -> None:
    case = EvaluationCase(
        question="签收后几天可以申请退货？",
        expected_answer_points=["签收后七天内"],
        expected_context_ids=["policy-returns-1"],
        category="after_sales",
    )

    result = score_case(
        case,
        answer="根据知识库：签收后七天内可以申请退货。",
        contexts=["签收后七天内可以申请退货。商品应保持完好。"],
        retrieved_ids=["policy-returns-1"],
    )

    assert set(result) == {
        "faithfulness",
        "answer_relevancy",
        "context_recall",
        "context_precision",
    }
    assert all(value == 1.0 for value in result.values())


def test_markdown_report_includes_model_mode_metadata() -> None:
    markdown = load_evaluate_rag_module().render_markdown(
        {
            "generated_at": "2026-05-27T00:00:00+00:00",
            "retrieval_mode": "milvus_hybrid",
            "model_mode": "rules",
            "embedding_model": "BAAI/bge-small-zh-v1.5",
            "case_count": 8,
            "metrics": {
                "faithfulness": 1.0,
                "answer_relevancy": 1.0,
                "context_recall": 1.0,
                "context_precision": 1.0,
            },
        }
    )

    assert "- Model mode: `rules`" in markdown


def test_markdown_report_only_mentions_offline_warning_for_offline_mode() -> None:
    markdown = load_evaluate_rag_module().render_markdown(
        {
            "generated_at": "2026-05-27T00:00:00+00:00",
            "retrieval_mode": "milvus_hybrid",
            "model_mode": "rules",
            "embedding_model": "BAAI/bge-small-zh-v1.5",
            "case_count": 8,
            "metrics": {
                "faithfulness": 1.0,
                "answer_relevancy": 1.0,
                "context_recall": 1.0,
                "context_precision": 1.0,
            },
        }
    )

    assert "offline_markdown_baseline" not in markdown

    offline_markdown = load_evaluate_rag_module().render_markdown(
        {
            "generated_at": "2026-05-27T00:00:00+00:00",
            "retrieval_mode": "offline_markdown_baseline",
            "model_mode": "rules",
            "embedding_model": "not_used",
            "case_count": 8,
            "metrics": {
                "faithfulness": 1.0,
                "answer_relevancy": 1.0,
                "context_recall": 1.0,
                "context_precision": 1.0,
            },
        }
    )

    assert "offline_markdown_baseline" in offline_markdown
