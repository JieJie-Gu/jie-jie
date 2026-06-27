# 生成客服 Agent 端到端评测专用数据，不影响最小 demo seed。
from __future__ import annotations

import argparse
from datetime import UTC, datetime, timedelta
import sys
from pathlib import Path
from typing import Any

from sqlalchemy import delete, func, select
from sqlalchemy.engine import make_url


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from smart_cs.config import Settings  # noqa: E402
from smart_cs.domain.enums import OrderStatus  # noqa: E402
from smart_cs.domain.models import (  # noqa: E402
    AgentRun,
    Conversation,
    ConversationSummary,
    Customer,
    Message,
    MemoryRecord,
    Order,
    PendingAction,
    Product,
    Ticket,
    ToolCall,
)
from smart_cs.infrastructure.database import Database  # noqa: E402
from smart_cs.infrastructure.repositories import SqlRepository  # noqa: E402


CUSTOMER_IDS = [f"C{index:04d}" for index in range(1001, 1051)]
PRODUCT_IDS = [f"P{index:04d}" for index in range(2001, 2031)]
ORDER_IDS = [f"O{index:06d}" for index in range(300001, 300101)]

PRODUCTS = [
    ("P2001", "黑色通勤轻便鞋", "黑色 通勤 轻量 日常 预算300元以内 适合上班步行", 26900),
    ("P2002", "轻量慢跑训练鞋", "轻量跑步鞋 平时慢跑 缓震 透气 适合5公里训练", 32900),
    ("P2003", "宽脚舒适休闲鞋", "宽脚友好 鞋楦偏宽 久站舒适 日常休闲", 29900),
    ("P2004", "雨天防滑通勤鞋", "防滑 雨天 通勤 耐磨橡胶底 黑色", 28900),
    ("P2005", "学生预算跑鞋", "学生预算 价格友好 300元以内 慢跑 轻量", 19900),
    ("P2006", "高缓震跑步鞋", "缓震 回弹 跑步 长距离训练 支撑", 45900),
    ("P2007", "白色百搭休闲鞋", "白色 百搭 休闲 通勤 简洁", 25900),
    ("P2008", "夏季透气网面鞋", "透气 网面 夏季 轻便 日常", 23900),
    ("P2009", "防泼水户外鞋", "防泼水 户外 防滑 耐磨 雨天", 39900),
    ("P2010", "软底健步鞋", "软底 健步 久走 防滑 轻便", 29900),
    ("P2011", "儿童魔术贴运动鞋", "儿童 魔术贴 运动 防滑 耐磨", 21900),
    ("P2012", "篮球缓震高帮鞋", "篮球 高帮 缓震 支撑 防扭", 49900),
    ("P2013", "开胶售后测试鞋 A", "售后测试 鞋底 开胶 通勤 delivered", 31900),
    ("P2014", "破损售后测试鞋 B", "售后测试 破损 鞋面 日常 delivered", 34900),
    ("P2015", "尺码退换测试鞋 C", "售后测试 尺码偏小 退换 宽脚", 29900),
    ("P2016", "脱线售后测试鞋 D", "售后测试 鞋面脱线 质量问题", 36900),
    ("P2017", "转人工测试鞋 E", "售后测试 质量争议 转人工", 42900),
    ("P2018", "鞋盒破损测试鞋 F", "售后测试 鞋盒破损 凭证 图片", 27900),
    ("P2019", "黑色宽脚通勤鞋", "黑色 宽脚 通勤 预算300元左右", 29900),
    ("P2020", "低预算日常鞋", "预算 200元以内 学生 日常 轻便", 18900),
    ("P2021", "窄脚包裹跑鞋", "窄脚 包裹 跑步 竞速", 52900),
    ("P2022", "灰色通勤休闲鞋", "灰色 通勤 休闲 百搭", 30900),
    ("P2023", "大码舒适鞋", "大码 42码 43码 宽脚 舒适", 32900),
    ("P2024", "深色防滑工作鞋", "深色 防滑 工作 长时间站立", 35900),
    ("P2025", "轻便旅行鞋", "旅行 轻便 透气 长走路", 33900),
    ("P2026", "冬季保暖鞋", "保暖 冬季 防滑 厚底", 38900),
    ("P2027", "低帮篮球训练鞋", "篮球 低帮 训练 缓震", 43900),
    ("P2028", "粉色休闲鞋", "粉色 休闲 女款 轻便", 26900),
    ("P2029", "耐磨工装鞋", "耐磨 工装 防滑 深色", 39900),
    ("P2030", "极简白色通勤鞋", "白色 通勤 极简 日常", 31900),
]

MEMORY_SPECS: dict[str, list[dict[str, Any]]] = {
    "C1005": [
        {
            "key": "color_preference",
            "memory_kind": "semantic",
            "memory_type": "preference",
            "title": "颜色偏好",
            "description": "用户偏好黑色和深色鞋款。",
            "value": {"attribute": "颜色", "value": "黑色"},
        }
    ],
    "C1006": [
        {
            "key": "budget_preference",
            "memory_kind": "semantic",
            "memory_type": "preference",
            "title": "预算偏好",
            "description": "用户通常希望鞋款预算在300元以内。",
            "value": {"attribute": "预算", "value": "300元以内"},
        }
    ],
    "C1019": [
        {
            "key": "shoe_size",
            "memory_kind": "semantic",
            "memory_type": "preference",
            "title": "尺码偏好",
            "description": "用户之前反馈常穿42码。",
            "value": {"attribute": "尺码", "value": "42码"},
        }
    ],
    "C1020": [
        {
            "key": "budget_preference",
            "memory_kind": "semantic",
            "memory_type": "preference",
            "title": "预算偏好",
            "description": "用户之前要求预算控制在300元以内。",
            "value": {"attribute": "预算", "value": "300元以内"},
        }
    ],
    "C1021": [
        {
            "key": "size_return_event",
            "memory_kind": "episodic",
            "memory_type": "after_sales_event",
            "title": "尺码退货历史",
            "description": "用户上次因为尺码偏小申请过退货。",
            "value": {"issue": "尺码偏小", "action_type": "after_sales", "status": "submitted"},
        }
    ],
    "C1022": [
        {
            "key": "color_preference",
            "memory_kind": "semantic",
            "memory_type": "preference",
            "title": "颜色偏好",
            "description": "用户之前喜欢黑色鞋款。",
            "value": {"attribute": "颜色", "value": "黑色"},
        }
    ],
    "C1023": [
        {
            "key": "wide_feet_preference",
            "memory_kind": "semantic",
            "memory_type": "preference",
            "title": "宽脚偏好",
            "description": "用户之前提到脚宽，偏好宽脚友好的鞋。",
            "value": {"attribute": "宽脚", "value": "宽脚鞋"},
        }
    ],
    "C1024": [
        {
            "key": "general_preference",
            "memory_kind": "semantic",
            "memory_type": "preference",
            "title": "综合偏好",
            "description": "用户偏好轻量、深色、预算300元左右的日常鞋。",
            "value": {"attribute": "偏好", "value": "轻量 深色 预算300元"},
        }
    ],
}


def ensure_sqlite_parent_directory(database_url: str) -> None:
    url = make_url(database_url)
    if url.get_backend_name() != "sqlite" or url.database in {None, "", ":memory:"}:
        return
    Path(url.database).expanduser().parent.mkdir(parents=True, exist_ok=True)


def seed_agent_eval_data(database_url: str | None = None, *, reset_eval: bool = False) -> dict[str, int]:
    settings = Settings()
    active_url = database_url or settings.database_url
    ensure_sqlite_parent_directory(active_url)
    database = Database(active_url)
    repository = SqlRepository(database)
    repository.create_schema()
    repository.seed_demo_data()
    try:
        with database.session() as session:
            if reset_eval:
                _reset_eval_data(session)
            _seed_customers(session)
            _seed_products(session)
            _seed_orders(session)
            _seed_conversations_and_messages(session)
            _seed_memories(session)
        return _summary(database)
    finally:
        database.dispose()


def _reset_eval_data(session) -> None:
    seeded_conversation_ids = [f"eval-conv-{customer_id}" for customer_id in CUSTOMER_IDS]
    eval_conversation_ids = list(
        session.scalars(
            select(Conversation.id).where(
                (Conversation.customer_id.in_(CUSTOMER_IDS))
                | (Conversation.id.in_(seeded_conversation_ids))
            )
        )
    )
    eval_action_ids = list(
        session.scalars(
            select(PendingAction.id).where(
                (PendingAction.customer_id.in_(CUSTOMER_IDS))
                | (PendingAction.conversation_id.in_(eval_conversation_ids))
                | (PendingAction.order_id.in_(ORDER_IDS))
            )
        )
    )
    if eval_action_ids:
        session.execute(delete(Ticket).where(Ticket.action_id.in_(eval_action_ids)))
    session.execute(
        delete(ToolCall).where(
            (ToolCall.customer_id.in_(CUSTOMER_IDS))
            | (ToolCall.arguments["conversation_id"].as_string().in_(eval_conversation_ids))
        )
    )
    session.execute(delete(AgentRun).where(AgentRun.conversation_id.in_(eval_conversation_ids)))
    if eval_action_ids:
        session.execute(delete(PendingAction).where(PendingAction.id.in_(eval_action_ids)))
    session.execute(
        delete(ConversationSummary).where(
            (ConversationSummary.conversation_id.in_(eval_conversation_ids))
            | (ConversationSummary.customer_id.in_(CUSTOMER_IDS))
        )
    )
    session.execute(
        delete(Message).where(
            (Message.conversation_id.in_(eval_conversation_ids))
            | (Message.customer_id.in_(CUSTOMER_IDS))
        )
    )
    session.execute(delete(Conversation).where(Conversation.id.in_(eval_conversation_ids)))
    session.execute(delete(MemoryRecord).where(MemoryRecord.owner_id.in_(CUSTOMER_IDS)))
    session.execute(delete(Order).where(Order.id.in_(ORDER_IDS)))
    session.execute(delete(Product).where(Product.id.in_(PRODUCT_IDS)))
    session.execute(delete(Customer).where(Customer.id.in_(CUSTOMER_IDS)))


def _seed_customers(session) -> None:
    personas = ["通勤", "跑步", "宽脚", "防滑", "预算", "售后", "记忆", "政策"]
    for index, customer_id in enumerate(CUSTOMER_IDS, start=1):
        name = f"{personas[(index - 1) % len(personas)]}评测客户{index:02d}"
        row = session.get(Customer, customer_id)
        if row is None:
            session.add(Customer(id=customer_id, name=name))
        else:
            row.name = name


def _seed_products(session) -> None:
    for product_id, name, description, price_cents in PRODUCTS:
        row = session.get(Product, product_id)
        if row is None:
            session.add(
                Product(
                    id=product_id,
                    name=name,
                    description=description,
                    price_cents=price_cents,
                    active=True,
                )
            )
        else:
            row.name = name
            row.description = description
            row.price_cents = price_cents
            row.active = True


def _seed_orders(session) -> None:
    product_prices = {product_id: price for product_id, _name, _description, price in PRODUCTS}
    for offset, order_id in enumerate(ORDER_IDS):
        customer_id = CUSTOMER_IDS[offset // 2]
        product_id = PRODUCT_IDS[offset % len(PRODUCT_IDS)]
        quantity = 1 if offset % 2 == 0 else 2
        row = session.get(Order, order_id)
        values = {
            "customer_id": customer_id,
            "product_id": product_id,
            "status": OrderStatus.DELIVERED.value,
            "quantity": quantity,
            "total_cents": product_prices[product_id] * quantity,
        }
        if row is None:
            session.add(Order(id=order_id, **values))
        else:
            for key, value in values.items():
                setattr(row, key, value)


def _seed_conversations_and_messages(session) -> None:
    for index, customer_id in enumerate(CUSTOMER_IDS, start=1):
        conversation_id = f"eval-conv-{customer_id}"
        if session.get(Conversation, conversation_id) is None:
            session.add(Conversation(id=conversation_id, customer_id=customer_id))
        has_messages = session.scalar(
            select(func.count(Message.id)).where(Message.conversation_id == conversation_id)
        )
        if has_messages:
            continue
        first_order_id = ORDER_IDS[(index - 1) * 2]
        session.add_all(
            [
                Message(
                    conversation_id=conversation_id,
                    customer_id=customer_id,
                    role="user",
                    content="我买的鞋有点问题，鞋底开胶了，想售后。",
                ),
                Message(
                    conversation_id=conversation_id,
                    customer_id=customer_id,
                    role="assistant",
                    content="请提供订单号和问题照片，我先帮你核对售后条件。",
                ),
                Message(
                    conversation_id=conversation_id,
                    customer_id=customer_id,
                    role="user",
                    content=f"订单号是 {first_order_id}。",
                ),
            ]
        )


def _seed_memories(session) -> None:
    for customer_id, specs in MEMORY_SPECS.items():
        for spec in specs:
            _upsert_memory(session, customer_id, spec, review_status="approved")
        _upsert_candidate_memory(session, customer_id, status="pending")
        _upsert_candidate_memory(session, customer_id, status="rejected")


def _upsert_memory(session, customer_id: str, spec: dict[str, Any], *, review_status: str) -> None:
    namespace = f"customer/{customer_id}/memories"
    memory_key = str(spec["key"])
    memory_id = f"{namespace}:{memory_key}"
    value_json = {
        "memory_kind": spec["memory_kind"],
        "memory_type": spec["memory_type"],
        "key": memory_key,
        "title": spec["title"],
        "description": spec["description"],
        "value": spec["value"],
        "review_status": review_status,
    }
    row = session.get(MemoryRecord, memory_id)
    values = {
        "namespace": namespace,
        "scope": "customer",
        "owner_id": customer_id,
        "memory_type": spec["memory_type"],
        "key": memory_key,
        "title": spec["title"],
        "description": spec["description"],
        "value_json": value_json,
        "evidence_json": [{"source": "seed_agent_eval_data"}],
        "source": "evaluation_seed",
        "confidence": "high",
        "risk_level": "low",
        "review_status": review_status,
        "created_by": "seed_agent_eval_data",
        "approved_by": "eval-seed" if review_status == "approved" else None,
        "expires_at": datetime.now(UTC) + timedelta(days=365),
    }
    if row is None:
        session.add(MemoryRecord(id=memory_id, **values))
    else:
        for key, value in values.items():
            setattr(row, key, value)


def _upsert_candidate_memory(session, customer_id: str, *, status: str) -> None:
    namespace = f"customer/{customer_id}/memory_candidates"
    memory_key = f"candidate_{status}"
    memory_id = f"{namespace}:{memory_key}"
    title = f"{status}候选记忆"
    description = "这条候选记忆不得进入评测上下文。"
    value_json = {
        "memory_kind": "semantic",
        "memory_type": "preference",
        "key": memory_key,
        "title": title,
        "description": description,
        "value": {"attribute": "候选", "value": status},
        "review_status": status,
    }
    row = session.get(MemoryRecord, memory_id)
    values = {
        "namespace": namespace,
        "scope": "customer",
        "owner_id": customer_id,
        "memory_type": "preference",
        "key": memory_key,
        "title": title,
        "description": description,
        "value_json": value_json,
        "evidence_json": [{"source": "seed_agent_eval_data"}],
        "source": "evaluation_seed",
        "confidence": "medium",
        "risk_level": "low",
        "review_status": status,
        "created_by": "seed_agent_eval_data",
        "approved_by": None,
        "expires_at": datetime.now(UTC) + timedelta(days=365),
    }
    if row is None:
        session.add(MemoryRecord(id=memory_id, **values))
    else:
        for key, value in values.items():
            setattr(row, key, value)


def _summary(database: Database) -> dict[str, int]:
    eval_conversation_ids = [f"eval-conv-{customer_id}" for customer_id in CUSTOMER_IDS]
    with database.session() as session:
        return {
            "customers": session.scalar(select(func.count(Customer.id)).where(Customer.id.in_(CUSTOMER_IDS))) or 0,
            "products": session.scalar(select(func.count(Product.id)).where(Product.id.in_(PRODUCT_IDS))) or 0,
            "orders": session.scalar(select(func.count(Order.id)).where(Order.id.in_(ORDER_IDS))) or 0,
            "conversations": session.scalar(
                select(func.count(Conversation.id)).where(Conversation.id.in_(eval_conversation_ids))
            )
            or 0,
            "approved_memories": session.scalar(
                select(func.count(MemoryRecord.id)).where(
                    MemoryRecord.owner_id.in_(CUSTOMER_IDS),
                    MemoryRecord.namespace.like("customer/%/memories"),
                    MemoryRecord.review_status == "approved",
                )
            )
            or 0,
            "memory_candidates": session.scalar(
                select(func.count(MemoryRecord.id)).where(
                    MemoryRecord.owner_id.in_(CUSTOMER_IDS),
                    MemoryRecord.namespace.like("customer/%/memory_candidates"),
                )
            )
            or 0,
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset-eval", action="store_true")
    args = parser.parse_args()
    summary = seed_agent_eval_data(reset_eval=args.reset_eval)
    print(f"Seeded agent evaluation data: {summary}")


if __name__ == "__main__":
    main()
