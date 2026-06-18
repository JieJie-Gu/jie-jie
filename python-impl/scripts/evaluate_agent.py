# 调用 FastAPI 后端运行客服 Agent 测评用例，并生成精简指标报告。
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any
from urllib import error, parse, request


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from smart_cs.evaluation.agent_metrics import (  # noqa: E402
    AgentEvalCase,
    AgentEvalObservation,
    score_agent_case,
    summarize_scores,
)


DEFAULT_CASES_PATH = ROOT / "data" / "evaluation" / "agent_cases.json"
DEFAULT_JSON_OUTPUT = ROOT / "data" / "evaluation" / "agent_eval_latest.json"
DEFAULT_MARKDOWN_OUTPUT = ROOT / "data" / "evaluation" / "agent_eval_latest.md"


class AgentEvalClient:
    def __init__(self, base_url: str, *, timeout: int = 120) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def create_conversation(self, customer_id: str) -> dict[str, Any]:
        return self._request("POST", "/api/conversations", {"customer_id": customer_id})

    def send_message(self, conversation_id: str, customer_id: str, content: str) -> dict[str, Any]:
        return self._request(
            "POST",
            f"/api/conversations/{conversation_id}/messages",
            {"customer_id": customer_id, "content": content},
        )

    def confirm(
        self,
        conversation_id: str,
        customer_id: str,
        action_id: str,
        *,
        approved: bool,
    ) -> dict[str, Any]:
        return self._request(
            "POST",
            f"/api/conversations/{conversation_id}/actions/confirm",
            {"customer_id": customer_id, "action_id": action_id, "approved": approved},
        )

    def list_tool_calls(self, conversation_id: str, customer_id: str) -> dict[str, Any]:
        return self._request(
            "GET",
            f"/api/conversations/{conversation_id}/tool-calls",
            query={"customer_id": customer_id},
        )

    def current_context(self, conversation_id: str, customer_id: str) -> dict[str, Any]:
        return self._request(
            "GET",
            f"/api/conversations/{conversation_id}/context",
            query={"customer_id": customer_id},
        )

    def _request(
        self,
        method: str,
        path: str,
        body: dict[str, Any] | None = None,
        *,
        query: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        url = self.base_url + path
        if query:
            url += "?" + parse.urlencode(query)
        data = None
        headers: dict[str, str] = {}
        if body is not None:
            data = json.dumps(body, ensure_ascii=False).encode("utf-8")
            headers["Content-Type"] = "application/json"
        req = request.Request(url, data=data, headers=headers, method=method)
        try:
            with request.urlopen(req, timeout=self.timeout) as response:
                payload = response.read().decode("utf-8")
        except error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"HTTP {exc.code} {url}: {detail}") from exc
        except error.URLError as exc:
            raise RuntimeError(f"Unable to call {url}: {exc}") from exc
        return json.loads(payload) if payload else {}


def load_cases(path: Path) -> list[AgentEvalCase]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    return [AgentEvalCase.from_dict(row) for row in rows]


def run_case(client: AgentEvalClient, case: AgentEvalCase) -> AgentEvalObservation:
    responses: list[dict[str, Any]] = []
    confirm_response: dict[str, Any] | None = None
    tool_calls: list[dict[str, Any]] = []
    context: dict[str, Any] = {}
    try:
        conversation = client.create_conversation(case.customer_id)
        conversation_id = str(conversation["id"])
        for message in case.messages:
            responses.append(client.send_message(conversation_id, case.customer_id, message))

        pending_action = _latest_pending_action(responses)
        if pending_action and case.confirm in {"approve", "reject"}:
            confirm_response = client.confirm(
                conversation_id,
                case.customer_id,
                str(pending_action["action_id"]),
                approved=case.confirm == "approve",
            )

        tool_calls = client.list_tool_calls(conversation_id, case.customer_id).get("tool_calls", [])
        context = client.current_context(conversation_id, case.customer_id)
        return AgentEvalObservation(
            responses=responses,
            confirm_response=confirm_response,
            tool_calls=tool_calls,
            context=context,
        )
    except Exception as exc:
        return AgentEvalObservation(
            responses=responses,
            confirm_response=confirm_response,
            tool_calls=tool_calls,
            context=context,
            error=str(exc),
        )


def build_report(
    *,
    cases: list[AgentEvalCase],
    observations: list[AgentEvalObservation],
    base_url: str,
) -> dict[str, Any]:
    scores = [score_agent_case(case, observation) for case, observation in zip(cases, observations)]
    summary = summarize_scores(scores)
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "base_url": base_url,
        "summary": summary,
        "cases": [score.as_dict() for score in scores],
        "tool_call_summary": _tool_call_summary(observations),
        "memory_summary": _memory_summary(observations),
        "rag_summary": _rag_summary(observations),
    }


def render_markdown(report: dict[str, Any]) -> str:
    summary = report["summary"]
    dimensions = summary["dimension_scores"]
    lines = [
        "# Agent Evaluation Results",
        "",
        f"- Generated at: `{report['generated_at']}`",
        f"- Base URL: `{report['base_url']}`",
        f"- Total score: **{summary['total_score']:.2f} / 100**",
        f"- Band: `{summary['band']}`",
        f"- Passed: `{summary['passed']}`",
        f"- Redline triggered: `{summary['redline_triggered']}`",
        "",
        "## Dimension Scores",
        "",
        "| Dimension | Score |",
        "| --- | ---: |",
    ]
    for dimension, score in dimensions.items():
        lines.append(f"| {dimension} | {score:.2f} |")

    lines.extend(
        [
            "",
            "## Redline Violations",
            "",
        ]
    )
    if summary["redline_violations"]:
        for item in summary["redline_violations"]:
            lines.append(f"- `{item['case_id']}`: {', '.join(item['violations'])}")
    else:
        lines.append("- None")

    lines.extend(["", "## Failed Cases", ""])
    failures = summary["failures"]
    if failures:
        for item in failures:
            lines.append(f"- `{item['case_id']}`: {', '.join(item['failures'])}")
    else:
        lines.append("- None")

    lines.extend(
        [
            "",
            "## ToolCall Summary",
            "",
            "```json",
            json.dumps(report["tool_call_summary"], ensure_ascii=False, indent=2),
            "```",
            "",
            "## Memory Summary",
            "",
            "```json",
            json.dumps(report["memory_summary"], ensure_ascii=False, indent=2),
            "```",
            "",
            "## RAG Summary",
            "",
            "```json",
            json.dumps(report["rag_summary"], ensure_ascii=False, indent=2),
            "```",
        ]
    )
    return "\n".join(lines) + "\n"


def write_report(report: dict[str, Any], json_path: Path, markdown_path: Path) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    markdown_path.write_text(render_markdown(report), encoding="utf-8")


def _latest_pending_action(responses: list[dict[str, Any]]) -> dict[str, Any] | None:
    for response in reversed(responses):
        pending = response.get("pending_action")
        if isinstance(pending, dict):
            return pending
    return None


def _tool_call_summary(observations: list[AgentEvalObservation]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for observation in observations:
        for call in observation.tool_calls:
            name = str(call.get("tool_name") or "")
            if not name:
                continue
            counts[name] = counts.get(name, 0) + 1
    return counts


def _memory_summary(observations: list[AgentEvalObservation]) -> dict[str, Any]:
    selected = 0
    recalled = 0
    for observation in observations:
        for call in observation.tool_calls:
            if call.get("tool_name") == "memory_select":
                selected += int((call.get("result") or {}).get("count") or 0)
            if call.get("tool_name") == "recall_memory":
                result = call.get("result") or {}
                recalled += len(result.get("long_term_memories") or result.get("memories") or [])
    return {"selected_memories": selected, "recalled_memories": recalled}


def _rag_summary(observations: list[AgentEvalObservation]) -> dict[str, Any]:
    citations = 0
    calls = 0
    for observation in observations:
        for call in observation.tool_calls:
            if call.get("tool_name") != "knowledge_rag":
                continue
            calls += 1
            citations += len((call.get("result") or {}).get("citations") or [])
    return {"knowledge_rag_calls": calls, "citations": citations}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES_PATH)
    parser.add_argument("--json-out", type=Path, default=DEFAULT_JSON_OUTPUT)
    parser.add_argument("--md-out", type=Path, default=DEFAULT_MARKDOWN_OUTPUT)
    args = parser.parse_args()

    cases = load_cases(args.cases)
    client = AgentEvalClient(args.base_url)
    observations = [run_case(client, case) for case in cases]
    report = build_report(cases=cases, observations=observations, base_url=args.base_url)
    write_report(report, args.json_out, args.md_out)
    print(f"Wrote {args.json_out} and {args.md_out}")


if __name__ == "__main__":
    main()
