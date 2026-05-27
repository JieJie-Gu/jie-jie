from __future__ import annotations

from dataclasses import dataclass
import re


METRIC_NAMES = (
    "faithfulness",
    "answer_relevancy",
    "context_recall",
    "context_precision",
)


@dataclass(frozen=True)
class EvaluationCase:
    question: str
    expected_answer_points: list[str]
    expected_context_ids: list[str]
    category: str


def score_contexts(case: EvaluationCase, retrieved_ids: list[str]) -> dict[str, float]:
    expected = set(case.expected_context_ids)
    retrieved = set(retrieved_ids)
    matched = expected & retrieved
    return {
        "context_recall": len(matched) / len(expected) if expected else 1.0,
        "context_precision": len(matched) / len(retrieved) if retrieved else 0.0,
    }


def score_case(
    case: EvaluationCase,
    *,
    answer: str,
    contexts: list[str],
    retrieved_ids: list[str],
) -> dict[str, float]:
    context_metrics = score_contexts(case, retrieved_ids)
    assertions = _answer_assertions(answer)
    joined_context = _normalize("".join(contexts))
    supported = sum(_normalize(assertion) in joined_context for assertion in assertions)
    answer_relevant = sum(_normalize(point) in _normalize(answer) for point in case.expected_answer_points)
    return {
        "faithfulness": supported / len(assertions) if assertions else 0.0,
        "answer_relevancy": (
            answer_relevant / len(case.expected_answer_points)
            if case.expected_answer_points
            else 1.0
        ),
        "context_recall": context_metrics["context_recall"],
        "context_precision": context_metrics["context_precision"],
    }


def average_metrics(case_metrics: list[dict[str, float]]) -> dict[str, float]:
    if not case_metrics:
        return {metric: 0.0 for metric in METRIC_NAMES}
    return {
        metric: round(sum(result[metric] for result in case_metrics) / len(case_metrics), 4)
        for metric in METRIC_NAMES
    }


def _answer_assertions(answer: str) -> list[str]:
    evidence_answer = answer.removeprefix("根据知识库：")
    return [
        assertion.strip()
        for assertion in re.split(r"[。！？!?]", evidence_answer)
        if assertion.strip()
    ]


def _normalize(text: str) -> str:
    return re.sub(r"\s+", "", text)
