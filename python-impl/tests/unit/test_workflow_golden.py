import json
from pathlib import Path

import pytest

from smart_cs.application.agent_runtime import AgentRuntime
from smart_cs.application.memory import MemoryWriteback
from smart_cs.infrastructure.database import Database
from smart_cs.infrastructure.model_factory import RulesDecisionModel
from smart_cs.infrastructure.repositories import SqlRepository
from smart_cs.tools.executor import AuthorizedToolExecutor
from tests.api.support import StaticKnowledgeAgent


CASES_PATH = Path(__file__).parents[1] / "golden" / "agent_workflow_cases.jsonl"
BADCASE_OUTPUT = Path(__file__).parents[1] / "golden" / "badcases.generated.jsonl"


def load_cases() -> list[dict]:
    return [
        json.loads(line)
        for line in CASES_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


@pytest.fixture(scope="session", autouse=True)
def clear_badcase_output() -> None:
    BADCASE_OUTPUT.unlink(missing_ok=True)


def test_golden_cases_count_is_20() -> None:
    assert len(load_cases()) == 20


@pytest.mark.parametrize("case", load_cases(), ids=lambda case: case["id"])
def test_agent_workflow_golden_cases(case, tmp_path) -> None:
    repository = SqlRepository(Database(f"sqlite:///{tmp_path / (case['id'] + '.db')}"))
    repository.create_schema()
    repository.seed_demo_data()
    runtime = AgentRuntime(
        executor=AuthorizedToolExecutor(repository),
        decision_model=RulesDecisionModel(),
        checkpoint_path=tmp_path / f"{case['id']}-checkpoints.db",
        knowledge_agent=StaticKnowledgeAgent(),
        memory_writeback=MemoryWriteback(repository=repository),
    )
    try:
        conversation_id = f"golden-{case['id']}"
        result = {}
        for turn in case["turns"]:
            result = runtime.invoke(conversation_id, case["customer_id"], turn)
        _assert_case(case, result)
    except Exception as error:
        _record_badcase(case, {"error": type(error).__name__, "message": str(error)})
        raise
    finally:
        runtime.close()


def _assert_case(case: dict, result: dict) -> None:
    failures = []
    if result.get("status") != case["expected_status"]:
        failures.append(f"status={result.get('status')!r}")
    if result.get("agents_invoked") != case["expected_agents"]:
        failures.append(f"agents_invoked={result.get('agents_invoked')!r}")
    reply = result.get("reply") or ""
    for expected in case.get("expected_reply_contains", []):
        if expected not in reply:
            failures.append(f"reply_missing={expected!r}")

    action = result.get("pending_confirmation") or result.get("result") or {}
    if expected_action_type := case.get("expected_action_type"):
        if action.get("action_type") != expected_action_type:
            failures.append(f"action_type={action.get('action_type')!r}")
    if expected_order_id := case.get("expected_order_id"):
        if action.get("order_id") != expected_order_id:
            failures.append(f"order_id={action.get('order_id')!r}")

    if failures:
        _record_badcase(case, {"result": result, "failures": failures})
        pytest.fail("; ".join(failures))


def _record_badcase(case: dict, payload: dict) -> None:
    BADCASE_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with BADCASE_OUTPUT.open("a", encoding="utf-8") as file:
        file.write(json.dumps({"case": case, **payload}, ensure_ascii=False) + "\n")
