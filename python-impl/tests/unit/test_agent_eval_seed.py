# 测试 Agent 端到端评测专用 seed 数据脚本。
from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

from sqlalchemy import select

from smart_cs.domain.models import Customer, MemoryRecord, Order, Product
from smart_cs.infrastructure.database import Database


def _load_seed_module() -> ModuleType:
    script_path = Path(__file__).parents[2] / "scripts" / "seed_agent_eval_data.py"
    spec = importlib.util.spec_from_file_location("seed_agent_eval_data", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_seed_agent_eval_data_creates_supported_eval_dataset(tmp_path: Path) -> None:
    module = _load_seed_module()
    database_url = f"sqlite:///{(tmp_path / 'agent_eval.db').as_posix()}"

    summary = module.seed_agent_eval_data(database_url, reset_eval=True)

    assert summary == {
        "customers": 50,
        "products": 30,
        "orders": 100,
        "conversations": 50,
        "approved_memories": 8,
        "memory_candidates": 16,
    }

    database = Database(database_url)
    try:
        with database.session() as session:
            assert session.get(Customer, "C001") is not None
            assert session.get(Product, "P1001") is not None
            assert session.get(Order, "O1001") is not None

            order = session.get(Order, "O300013")
            assert order is not None
            assert order.customer_id == "C1007"
            assert order.product_id == "P2013"
            assert order.status == "delivered"

            product = session.get(Product, "P2001")
            assert product is not None
            assert "黑色" in product.description
            assert "通勤" in product.description
            assert "300" in product.description
    finally:
        database.dispose()


def test_seed_agent_eval_data_is_idempotent_and_keeps_candidates_separate(tmp_path: Path) -> None:
    module = _load_seed_module()
    database_url = f"sqlite:///{(tmp_path / 'agent_eval.db').as_posix()}"

    first = module.seed_agent_eval_data(database_url, reset_eval=True)
    second = module.seed_agent_eval_data(database_url, reset_eval=False)

    assert second == first

    database = Database(database_url)
    try:
        with database.session() as session:
            active = list(
                session.scalars(
                    select(MemoryRecord).where(
                        MemoryRecord.owner_id == "C1019",
                        MemoryRecord.namespace == "customer/C1019/memories",
                    )
                )
            )
            candidates = list(
                session.scalars(
                    select(MemoryRecord).where(
                        MemoryRecord.owner_id == "C1019",
                        MemoryRecord.namespace == "customer/C1019/memory_candidates",
                    )
                )
            )
            assert len(active) == 1
            assert active[0].review_status == "approved"
            assert {candidate.review_status for candidate in candidates} == {"pending", "rejected"}
    finally:
        database.dispose()


def test_reset_eval_only_removes_evaluation_id_ranges(tmp_path: Path) -> None:
    module = _load_seed_module()
    database_url = f"sqlite:///{(tmp_path / 'agent_eval.db').as_posix()}"

    module.seed_agent_eval_data(database_url, reset_eval=True)
    database = Database(database_url)
    try:
        with database.session() as session:
            session.add(Customer(id="C2000", name="非评测客户"))
            session.add(Product(id="P9999", name="非评测商品", description="不要被 reset 删除", price_cents=1))
            session.flush()
            session.add(
                Order(
                    id="O999999",
                    customer_id="C2000",
                    product_id="P9999",
                    status="delivered",
                    quantity=1,
                    total_cents=1,
                )
            )

        module.seed_agent_eval_data(database_url, reset_eval=True)

        with database.session() as session:
            assert session.get(Customer, "C2000") is not None
            assert session.get(Product, "P9999") is not None
            assert session.get(Order, "O999999") is not None
            assert session.scalar(select(MemoryRecord).where(MemoryRecord.owner_id == "C1005")) is not None
    finally:
        database.dispose()
