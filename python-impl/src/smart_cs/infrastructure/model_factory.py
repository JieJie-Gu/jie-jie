from __future__ import annotations

import re
from typing import Any

from langchain_openai import ChatOpenAI

from smart_cs.agents.state import RouteAnalysis, RouterContext, SupervisorContext, SupervisorDecision
from smart_cs.config import Settings
from smart_cs.infrastructure.prompts import ROUTER_PROMPT, SUPERVISOR_PROMPT


ORDER_ID_PATTERN = re.compile(r"(O\d+)", re.IGNORECASE)


class RulesDecisionModel:
    """Deterministic keyword routing for local operation without an LLM."""

    _after_sales_keywords = ("售后", "退款", "退货", "换货", "开胶", "破损", "质量问题")
    _handoff_keywords = ("转人工", "人工客服", "投诉", "人工处理")
    _product_keywords = ("商品", "产品", "推荐", "价格", "跑鞋")
    _order_keywords = ("订单", "物流", "发货", "收货", "配送")
    _knowledge_domain_keywords = ("退货", "退款", "售后", "换货", "物流", "发货", "配送", "运费", "保养", "尺码")
    _knowledge_question_keywords = ("规则", "政策", "多久", "几天", "期限", "怎么", "如何", "说明", "什么")

    def route(self, context: RouterContext | str) -> RouteAnalysis:
        if isinstance(context, str):
            context = RouterContext(current_message=context)
        message = context.current_message
        entities: dict[str, str] = {}
        order_match = ORDER_ID_PATTERN.search(message)
        if order_match is not None:
            entities["order_id"] = order_match.group(1).upper()
        turn_type = self._infer_turn_type(message, context)

        if self._contains(message, self._handoff_keywords):
            return RouteAnalysis(intent="handoff", entities=entities, risk="high", turn_type=turn_type)
        if self._contains(message, self._knowledge_domain_keywords) and self._contains(
            message, self._knowledge_question_keywords
        ):
            return RouteAnalysis(intent="knowledge", entities=entities, turn_type=turn_type)
        if self._contains(message, self._after_sales_keywords):
            missing = [] if entities.get("order_id") or context.conversation_slots.active_order_id else ["order_id"]
            return RouteAnalysis(
                intent="after_sales",
                entities=entities,
                risk="medium",
                turn_type=turn_type,
                missing_entities=missing,
            )
        if self._contains(message, self._product_keywords):
            return RouteAnalysis(intent="product", entities=entities, turn_type=turn_type)
        if entities or self._contains(message, self._order_keywords):
            return RouteAnalysis(intent="order", entities=entities, turn_type=turn_type)
        return RouteAnalysis(intent="knowledge", entities=entities, turn_type=turn_type)

    def plan(self, context: SupervisorContext) -> SupervisorDecision:
        route = context.route
        if route.intent == "after_sales":
            return SupervisorDecision(
                agents=["OrderAgent", "KnowledgeAgent", "AfterSalesAgent"],
                action="draft_after_sales",
                requires_confirmation=True,
                planning_flags=["requires_order_fact", "requires_policy_check"],
            )
        if route.intent == "handoff":
            return SupervisorDecision(
                agents=["HandoffAgent"], action="draft_handoff", requires_confirmation=True
            )
        if route.intent == "product":
            return SupervisorDecision(agents=["ProductAgent"], action="read")
        if route.intent == "order":
            return SupervisorDecision(agents=["OrderAgent"], action="read")
        return SupervisorDecision(agents=["KnowledgeAgent"], action="read")

    @staticmethod
    def _contains(message: str, keywords: tuple[str, ...]) -> bool:
        return any(keyword in message for keyword in keywords)

    @staticmethod
    def _infer_turn_type(message: str, context: RouterContext) -> str:
        stripped = message.strip()
        if stripped in {"确认", "可以", "提交", "提交吧", "确认提交"}:
            return "confirmation_like"
        if stripped in {"取消", "不用了", "先不要", "不提交"}:
            return "rejection_like"
        if any(marker in message for marker in ("不是", "说错", "改成")):
            return "correction"
        if context.conversation_slots.active_order_id and any(
            marker in message for marker in ("那", "这个", "刚才", "它", "这单")
        ):
            return "follow_up"
        return "new_request"


class LangChainDecisionModel:
    """Structured-output adapter for a configured LangChain chat model."""

    def __init__(self, chat_model: Any) -> None:
        self._routing_model = chat_model.with_structured_output(RouteAnalysis)
        self._planning_model = chat_model.with_structured_output(SupervisorDecision)

    def route(self, context: RouterContext | str) -> RouteAnalysis:
        if isinstance(context, str):
            context = RouterContext(current_message=context)
        result = self._routing_model.invoke(
            ROUTER_PROMPT.invoke({"context_json": context.model_dump_json()})
        )
        return RouteAnalysis.model_validate(result)

    def plan(self, context: SupervisorContext) -> SupervisorDecision:
        result = self._planning_model.invoke(
            SUPERVISOR_PROMPT.invoke({"context_json": context.model_dump_json()})
        )
        return SupervisorDecision.model_validate(result)


def configured_chat_model(settings: Settings) -> ChatOpenAI:
    return ChatOpenAI(
        model=settings.llm_model,
        api_key=settings.llm_api_key,
        base_url=settings.llm_base_url,
        temperature=0,
    )
