# 测试真实 API 评测脚本的时序快照、记忆统计和安全摘要。
from __future__ import annotations

from scripts.evaluate_agent import _memory_summary, build_report, run_case
from smart_cs.evaluation.agent_metrics import AgentEvalCase, AgentEvalObservation


class RecordingEvalClient:
    def __init__(self) -> None:
        self.tool_calls = [
            {
                "id": "TC1",
                "tool_name": "draft_after_sales",
                "result": {"status": "pending_confirmation"},
            }
        ]

    def create_conversation(self, customer_id: str):
        return {"id": "conv-1", "customer_id": customer_id}

    def send_message(self, conversation_id: str, customer_id: str, content: str):
        return {
            "status": "pending_confirmation",
            "reply": "请确认。",
            "pending_action": {"action_id": "A1", "status": "pending_confirmation"},
        }

    def confirm(self, conversation_id: str, customer_id: str, action_id: str, *, approved: bool):
        self.tool_calls.append(
            {
                "id": "TC2",
                "tool_name": "submit_confirmed_action",
                "result": {"ticket_id": "T1"},
            }
        )
        return {"status": "completed", "result": {"ticket_id": "T1"}}

    def list_tool_calls(self, conversation_id: str, customer_id: str):
        return {"tool_calls": list(self.tool_calls)}

    def current_context(self, conversation_id: str, customer_id: str):
        return {"context": {"customer_memories": []}}


def test_run_case_captures_pre_confirm_and_post_confirm_delta() -> None:
    observation = run_case(
        RecordingEvalClient(),
        AgentEvalCase(
            case_id="after_sales_001",
            customer_id="C1001",
            messages=["申请售后"],
            confirm="approve",
        ),
    )

    assert [call["id"] for call in observation.pre_confirm_tool_calls] == ["TC1"]
    assert [call["id"] for call in observation.post_confirm_tool_calls] == ["TC2"]
    assert [call["id"] for call in observation.tool_calls] == ["TC1", "TC2"]


def test_memory_summary_counts_current_long_term_result_shape() -> None:
    summary = _memory_summary(
        [
            AgentEvalObservation(
                tool_calls=[
                    {"tool_name": "memory_select", "result": {"count": 2}},
                    {
                        "tool_name": "recall_memory",
                        "result": {
                            "long_term": {
                                "semantic_memories": [{"memory_id": "M1"}],
                                "episodic_memories": [{"memory_id": "M2"}],
                            }
                        },
                    },
                ]
            )
        ]
    )

    assert summary == {"selected_memories": 2, "recalled_memories": 2}


def test_report_contains_safe_case_trace_without_raw_memory_payload() -> None:
    case = AgentEvalCase(case_id="memory_001", customer_id="C1001", messages=["我的尺码？"])
    observation = AgentEvalObservation(
        responses=[{"status": "completed", "reply": "42 码"}],
        pre_confirm_tool_calls=[
            {
                "tool_name": "memory_select",
                "status": "succeeded",
                "result": {"memories": [{"value": {"shoe_size": 42}}]},
            }
        ],
        tool_calls=[
            {
                "tool_name": "memory_select",
                "status": "succeeded",
                "result": {"memories": [{"value": {"shoe_size": 42}}]},
            }
        ],
    )

    report = build_report(cases=[case], observations=[observation], base_url="http://test")
    trace = report["case_traces"][0]

    assert trace["case_id"] == "memory_001"
    assert trace["response_statuses"] == ["completed"]
    assert trace["pre_confirm_tools"] == [
        {"tool_name": "memory_select", "status": "succeeded", "error_type": None}
    ]
    assert "value" not in str(trace)
