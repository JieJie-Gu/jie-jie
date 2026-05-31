# 测试长期记忆复用向量检索底座后的召回、回表和安全投影。
from __future__ import annotations

from datetime import UTC, datetime, timedelta

from langchain_core.documents import Document

from smart_cs.application.memory_retrieval import (
    MemoryRetrievalService,
    MemoryVectorIndex,
    is_indexable_memory,
    memory_index_document,
)


def approved_memory(**overrides):
    value = {
        "memory_id": "customer/C001/memories:preference:shoe_size",
        "namespace": "customer/C001/memories",
        "key": "preference:shoe_size",
        "owner_id": "C001",
        "memory_kind": "semantic",
        "memory_type": "preference",
        "title": "鞋码偏好",
        "description": "用户通常穿 42 码。",
        "value": {"shoe_size": "42"},
        "confidence": "high",
        "risk_level": "low",
        "review_status": "approved",
        "updated_at": datetime.now(UTC).isoformat(),
    }
    value.update(overrides)
    return value


class FakeMemoryStore:
    def __init__(self, records):
        self.records = records
        self.gets = []
        self.searches = []

    def get(self, namespace, key):
        self.gets.append((namespace, key))
        return self.records.get((namespace, key))

    def search(self, namespace, query: str, limit: int):
        self.searches.append((namespace, query, limit))
        return [
            value
            for (stored_namespace, _key), value in self.records.items()
            if stored_namespace == namespace
        ][:limit]


class FakeVectorStore:
    def __init__(self, documents=None, *, fail_search: bool = False):
        self.documents = documents or []
        self.fail_search = fail_search
        self.added = []
        self.deleted = []
        self.search_kwargs = []

    def add_documents(self, documents, **kwargs):
        self.added.append((documents, kwargs))
        return kwargs.get("ids") or []

    def delete(self, ids=None, expr=None, **kwargs):
        self.deleted.append({"ids": ids, "expr": expr, "kwargs": kwargs})
        return True

    def similarity_search(self, query: str, **kwargs):
        if self.fail_search:
            raise RuntimeError("vector down")
        self.search_kwargs.append({"query": query, **kwargs})
        return self.documents


def test_memory_index_document_contains_search_text_and_pointer_metadata() -> None:
    document = memory_index_document(approved_memory())

    assert "鞋码偏好" in document.page_content
    assert "用户通常穿 42 码" in document.page_content
    assert "shoe_size" in document.page_content
    assert document.metadata["memory_id"] == "customer/C001/memories:preference:shoe_size"
    assert document.metadata["key"] == "preference:shoe_size"
    assert document.metadata["customer_id"] == "C001"
    assert document.metadata["namespace"] == "customer/C001/memories"


def test_memory_vector_index_only_upserts_indexable_memories() -> None:
    store = FakeVectorStore()
    index = MemoryVectorIndex(store)

    index.sync_record(approved_memory())
    index.sync_record(approved_memory(review_status="pending"))

    assert len(store.added) == 1
    assert store.added[0][1]["ids"] == ["customer/C001/memories:preference:shoe_size"]
    assert len(store.deleted) == 2
    assert is_indexable_memory(approved_memory()) is True
    assert is_indexable_memory(approved_memory(review_status="pending")) is False
    assert is_indexable_memory(approved_memory(risk_level="high")) is False
    assert is_indexable_memory(
        approved_memory(expires_at=(datetime.now(UTC) - timedelta(days=1)).isoformat())
    ) is False


def test_vector_search_hydrates_sql_and_filters_unsafe_records() -> None:
    vector = FakeVectorStore(
        [
            Document(page_content="", metadata={"key": "preference:shoe_size"}),
            Document(page_content="", metadata={"key": "candidate"}),
            Document(page_content="", metadata={"key": "preference:other_customer"}),
        ]
    )
    namespace = ("customer", "C001", "memories")
    store = FakeMemoryStore(
        {
            (namespace, "preference:shoe_size"): approved_memory(),
            (namespace, "candidate"): approved_memory(
                key="candidate",
                memory_id="customer/C001/memories:candidate",
                review_status="pending",
            ),
            (("customer", "C002", "memories"), "preference:other_customer"): approved_memory(
                memory_id="customer/C002/memories:preference:other_customer",
                namespace="customer/C002/memories",
                key="preference:other_customer",
                owner_id="C002",
            ),
        }
    )
    service = MemoryRetrievalService(store, vector_index=MemoryVectorIndex(vector))

    result = service.search_active_memories(
        customer_id="C001",
        query="我之前穿多大的",
        intent="pre_sales",
        limit=5,
    )

    assert [memory["memory_id"] for memory in result] == [
        "customer/C001/memories:preference:shoe_size"
    ]
    assert store.gets == [
        (namespace, "preference:shoe_size"),
        (namespace, "candidate"),
        (namespace, "preference:other_customer"),
    ]
    assert "customer_id == \"C001\"" in vector.search_kwargs[0]["expr"]
    assert "memory_candidates" not in vector.search_kwargs[0]["expr"]
    assert "value" not in result[0]


def test_vector_failure_falls_back_to_sql_search() -> None:
    namespace = ("customer", "C001", "memories")
    store = FakeMemoryStore({(namespace, "preference:shoe_size"): approved_memory()})
    service = MemoryRetrievalService(
        store,
        vector_index=MemoryVectorIndex(FakeVectorStore(fail_search=True)),
    )

    result = service.search_active_memories(
        customer_id="C001",
        query="鞋码",
        intent="pre_sales",
        limit=5,
    )

    assert result[0]["memory_id"] == "customer/C001/memories:preference:shoe_size"
    assert store.searches == [(namespace, "鞋码", 20)]


def test_stale_vector_hits_fall_back_to_sql_when_selector_filters_all() -> None:
    namespace = ("customer", "C001", "memories")
    vector = FakeVectorStore([Document(page_content="", metadata={"key": "candidate"})])
    store = FakeMemoryStore(
        {
            (namespace, "candidate"): approved_memory(
                key="candidate",
                memory_id="customer/C001/memories:candidate",
                review_status="pending",
            ),
            (namespace, "preference:shoe_size"): approved_memory(),
        }
    )
    service = MemoryRetrievalService(store, vector_index=MemoryVectorIndex(vector))

    result = service.search_active_memories(
        customer_id="C001",
        query="鞋码",
        intent="pre_sales",
        limit=5,
    )

    assert result[0]["memory_id"] == "customer/C001/memories:preference:shoe_size"
    assert store.searches == [(namespace, "鞋码", 20)]
