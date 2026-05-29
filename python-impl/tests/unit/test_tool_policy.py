import pytest

from smart_cs.domain.errors import ToolPermissionError
from smart_cs.infrastructure.database import Database
from smart_cs.infrastructure.repositories import SqlRepository
from smart_cs.tools.executor import AuthorizedToolExecutor
from smart_cs.tools.policy import default_tool_registry


def test_tool_registry_policy_is_permission_metadata_only() -> None:
    policy = default_tool_registry().get("draft_after_sales")

    assert policy.allowed_agents == frozenset({"AfterSalesAgent"})
    assert policy.requires_confirmation is True
    assert not hasattr(policy, "args_schema")


def test_executor_rejects_wrong_caller_agent(tmp_path) -> None:
    repository = SqlRepository(Database(f"sqlite:///{tmp_path / 'policy.db'}"))
    repository.create_schema()
    repository.seed_demo_data()
    executor = AuthorizedToolExecutor(repository)

    with pytest.raises(ToolPermissionError):
        executor.invoke(
            "draft_after_sales",
            {"customer_id": "C001", "order_id": "O1001", "reason": "鞋底开胶"},
            caller_agent="OrderAgent",
        )
