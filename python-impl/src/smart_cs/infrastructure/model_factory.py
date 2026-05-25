from __future__ import annotations

import re
from typing import Any

from langchain_openai import ChatOpenAI

from smart_cs.agents.state import RouteAnalysis, SupervisorDecision
from smart_cs.config import Settings


ORDER_ID_PATTERN = re.compile(r"(O\d+)", re.IGNORECASE)


class RulesDecisionModel:
    """Deterministic keyword routing for local operation without an LLM."""

    _after_sales_keywords = ("售后", "退款", "退货", "换货", "开胶", "破损", "质量问题")
    _handoff_keywords = ("转人工", "人工客服", "投诉", "人工处理")
    _product_keywords = ("商品", "产品", "推荐", "价格", "跑鞋")
    _order_keywords = ("订单", "物流", "发货", "收货", "配送")

    def route(self, message: str) -> RouteAnalysis:
        entities: dict[str, str] = {}
        order_match = ORDER_ID_PATTERN.search(message)
        if order_match is not None:
            entities["order_id"] = order_match.group(1).upper()

        if self._contains(message, self._handoff_keywords):
            return RouteAnalysis(intent="handoff", entities=entities, risk="high")
        if self._contains(message, self._after_sales_keywords):
            return RouteAnalysis(intent="after_sales", entities=entities, risk="medium")
        if self._contains(message, self._product_keywords):
            return RouteAnalysis(intent="product", entities=entities)
        if entities or self._contains(message, self._order_keywords):
            return RouteAnalysis(intent="order", entities=entities)
        return RouteAnalysis(intent="knowledge", entities=entities)

    def plan(self, _message: str, route: RouteAnalysis) -> SupervisorDecision:
        if route.intent == "after_sales":
            return SupervisorDecision(
                agents=["OrderAgent", "AfterSalesAgent"], action="draft_after_sales"
            )
        if route.intent == "handoff":
            return SupervisorDecision(agents=["HandoffAgent"], action="draft_handoff")
        if route.intent == "product":
            return SupervisorDecision(agents=["ProductAgent"], action="read")
        if route.intent == "order":
            return SupervisorDecision(agents=["OrderAgent"], action="read")
        return SupervisorDecision(agents=["KnowledgeAgent"], action="read")

    @staticmethod
    def _contains(message: str, keywords: tuple[str, ...]) -> bool:
        return any(keyword in message for keyword in keywords)


class LangChainDecisionModel:
    """Structured-output adapter for a configured LangChain chat model."""

    def __init__(self, chat_model: Any) -> None:
        self._routing_model = chat_model.with_structured_output(RouteAnalysis)
        self._planning_model = chat_model.with_structured_output(SupervisorDecision)

    def route(self, message: str) -> RouteAnalysis:
        result = self._routing_model.invoke(
            "分析客户消息意图、实体与风险，不选择或授权任何工具。\n"
            f"客户消息：{message}"
        )
        return RouteAnalysis.model_validate(result)

    def plan(self, message: str, route: RouteAnalysis) -> SupervisorDecision:
        result = self._planning_model.invoke(
            "根据路由结果规划 specialist 执行顺序与动作；写动作将由系统强制确认。\n"
            f"客户消息：{message}\n路由结果：{route.model_dump_json()}"
        )
        return SupervisorDecision.model_validate(result)


def configured_chat_model(settings: Settings) -> ChatOpenAI:
    return ChatOpenAI(
        model=settings.llm_model,
        api_key=settings.llm_api_key,
        base_url=settings.llm_base_url,
        temperature=0,
    )
