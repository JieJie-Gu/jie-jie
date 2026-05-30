# 定义业务工具风险等级和允许调用方策略。

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class ToolPolicy:
    name: str
    risk_level: Literal["low", "medium", "high"]
    allowed_agents: frozenset[str]
    requires_confirmation: bool
    idempotent: bool


class ToolRegistry:
    def __init__(self, policies: list[ToolPolicy]) -> None:
        self._policies = {policy.name: policy for policy in policies}

    def get(self, name: str) -> ToolPolicy:
        try:
            return self._policies[name]
        except KeyError as error:
            raise ValueError(f"Unknown customer tool: {name}") from error

    def as_view(self) -> list[dict[str, object]]:
        return [
            {
                "name": policy.name,
                "risk_level": policy.risk_level,
                "allowed_agents": sorted(policy.allowed_agents),
                "requires_confirmation": policy.requires_confirmation,
                "idempotent": policy.idempotent,
            }
            for policy in self._policies.values()
        ]


def default_tool_registry() -> ToolRegistry:
    return ToolRegistry(
        [
            ToolPolicy(
                "search_products",
                "low",
                frozenset({"PreSalesAgent", "PostSalesAgent"}),
                False,
                True,
            ),
            ToolPolicy("lookup_order", "medium", frozenset({"PostSalesAgent"}), False, True),
            ToolPolicy(
                "knowledge_rag",
                "low",
                frozenset({"PreSalesAgent", "PostSalesAgent"}),
                False,
                True,
            ),
            ToolPolicy(
                "recall_memory",
                "low",
                frozenset({"PreSalesAgent", "PostSalesAgent"}),
                False,
                True,
            ),
            ToolPolicy(
                "request_after_sales",
                "high",
                frozenset({"PostSalesAgent"}),
                True,
                True,
            ),
            ToolPolicy("request_handoff", "medium", frozenset({"PostSalesAgent"}), True, True),
            ToolPolicy("draft_after_sales", "high", frozenset({"PostSalesAgent"}), True, True),
            ToolPolicy("draft_handoff", "medium", frozenset({"PostSalesAgent"}), True, True),
            ToolPolicy("submit_confirmed_action", "high", frozenset({"ConfirmActionNode"}), True, True),
            ToolPolicy("cancel_pending_action", "medium", frozenset({"ConfirmActionNode"}), False, True),
        ]
    )


def default_tool_policy_view() -> list[dict[str, object]]:
    return default_tool_registry().as_view()
