from smart_cs.rag.evaluation import EvaluationCase, score_case, score_contexts


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
