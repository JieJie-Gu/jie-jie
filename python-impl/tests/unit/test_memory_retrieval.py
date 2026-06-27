# 测试长期记忆复用向量检索底座后的召回、回表和安全投影。
from __future__ import annotations

from datetime import UTC, datetime, timedelta

from langchain_core.documents import Document
import pytest

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
        "title": "Shoe size preference",
        "description": "User usually wears size 42.",
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
        self.get_by_ids = []
        self.searches = []
        self.repository = None

    def get(self, namespace, key):
        self.gets.append((namespace, key))
        return self.records.get((namespace, key))

    def get_by_id(self, memory_id: str):
        self.get_by_ids.append(memory_id)
        for value in self.records.values():
            if value.get("memory_id") == memory_id or value.get("id") == memory_id:
                return value
        return None

    def search(self, namespace, query: str, limit: int):
        self.searches.append((namespace, query, limit))
        return [
            value
            for (stored_namespace, _key), value in self.records.items()
            if stored_namespace == namespace
        ][:limit]


class FakeVectorStore:
    def __init__(
        self,
        documents=None,
        *,
        fail_search: bool = False,
        fail_delete: bool = False,
        collection_exists: bool | None = None,
    ):
        self.documents = documents or []
        self.fail_search = fail_search
        self.fail_delete = fail_delete
        self.added = []
        self.deleted = []
        self.search_kwargs = []
        self._collection_exists = collection_exists

    def collection_exists(self):
        return self._collection_exists

    def add_documents(self, documents, **kwargs):
        self.added.append((documents, kwargs))
        return kwargs.get("ids") or []

    def delete(self, ids=None, expr=None, **kwargs):
        if self.fail_delete:
            raise RuntimeError("delete failed")
        self.deleted.append({"ids": ids, "expr": expr, "kwargs": kwargs})
        return True

    def similarity_search(self, query: str, **kwargs):
        if self.fail_search:
            raise RuntimeError("vector down")
        self.search_kwargs.append({"query": query, **kwargs})
        return self.documents


def test_memory_index_document_contains_compact_search_text_and_pointer_metadata() -> None:
    document = memory_index_document(approved_memory())

    assert "Shoe size preference" in document.page_content
    assert "User usually wears size 42." in document.page_content
    assert "attribute: shoe_size; value: 42" in document.page_content
    assert document.metadata["memory_id"] == "customer/C001/memories:preference:shoe_size"
    assert document.metadata["key"] == "preference:shoe_size"
    assert document.metadata["customer_id"] == "C001"
    assert document.metadata["namespace"] == "customer/C001/memories"
    assert document.metadata["risk_level"] == "low"
    assert document.metadata["confidence"] == "high"


def test_memory_index_document_does_not_store_forbidden_value_payload() -> None:
    document = memory_index_document(
        approved_memory(
            value={
                "shoe_size": "42",
                "business_result": {"secret": "do-not-index"},
                "evidence": [{"text": "do-not-index"}],
            }
        )
    )

    assert "secret" not in document.page_content
    assert "business_result" not in document.page_content
    assert "evidence" not in document.page_content


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
    assert is_indexable_memory(approved_memory(confidence="low")) is False
    assert is_indexable_memory(approved_memory(namespace="customer/C001/memory_candidates")) is False
    assert is_indexable_memory(
        approved_memory(expires_at=(datetime.now(UTC) - timedelta(days=1)).isoformat())
    ) is False


def test_vector_search_hydrates_by_memory_id_and_filters_unsafe_records() -> None:
    vector = FakeVectorStore(
        [
            Document(
                page_content="",
                metadata={"memory_id": "customer/C001/memories:preference:shoe_size"},
            ),
            Document(page_content="", metadata={"key": "candidate"}),
            Document(
                page_content="",
                metadata={"memory_id": "customer/C002/memories:preference:other_customer"},
            ),
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
        query="shoe size",
        intent="pre_sales",
        limit=5,
    )

    assert [memory["memory_id"] for memory in result] == [
        "customer/C001/memories:preference:shoe_size"
    ]
    assert store.get_by_ids == [
        "customer/C001/memories:preference:shoe_size",
        "customer/C002/memories:preference:other_customer",
    ]
    assert store.gets == [(namespace, "candidate")]
    assert "customer_id == \"C001\"" in vector.search_kwargs[0]["expr"]
    assert "risk_level != \"high\"" in vector.search_kwargs[0]["expr"]
    assert "confidence != \"low\"" in vector.search_kwargs[0]["expr"]
    assert "memory_type != \"sensitive_label\"" in vector.search_kwargs[0]["expr"]
    assert "value" not in result[0]


def test_vector_and_sql_results_are_always_merged() -> None:
    namespace = ("customer", "C001", "memories")
    vector = FakeVectorStore(
        [
            Document(
                page_content="",
                metadata={"memory_id": "customer/C001/memories:episode:after_sales"},
            )
        ]
    )
    store = FakeMemoryStore(
        {
            (namespace, "episode:after_sales"): approved_memory(
                memory_id="customer/C001/memories:episode:after_sales",
                key="episode:after_sales",
                memory_kind="episodic",
                memory_type="after_sales_event",
                title="After-sales event",
                description="User submitted after-sales request for order O1001.",
                value={"order_id": "O1001", "action_type": "after_sales", "status": "submitted"},
            ),
            (namespace, "preference:shoe_size"): approved_memory(),
        }
    )
    service = MemoryRetrievalService(store, vector_index=MemoryVectorIndex(vector))

    result = service.search_active_memories(
        customer_id="C001",
        query="shoe size",
        intent="pre_sales",
        limit=5,
    )

    assert {memory["memory_id"] for memory in result} == {
        "customer/C001/memories:episode:after_sales",
        "customer/C001/memories:preference:shoe_size",
    }
    assert store.searches == [(namespace, "shoe size", 20)]


def test_vector_failure_falls_back_to_sql_search() -> None:
    namespace = ("customer", "C001", "memories")
    store = FakeMemoryStore({(namespace, "preference:shoe_size"): approved_memory()})
    service = MemoryRetrievalService(
        store,
        vector_index=MemoryVectorIndex(FakeVectorStore(fail_search=True)),
    )

    result = service.search_active_memories(
        customer_id="C001",
        query="shoe size",
        intent="pre_sales",
        limit=5,
    )

    assert result[0]["memory_id"] == "customer/C001/memories:preference:shoe_size"
    assert store.searches == [(namespace, "shoe size", 20)]


def test_upsert_raises_when_stale_delete_fails() -> None:
    index = MemoryVectorIndex(FakeVectorStore(fail_delete=True))

    try:
        index.upsert(approved_memory())
    except RuntimeError as error:
        assert "delete failed" in str(error) or "stale memory vector" in str(error)
    else:
        raise AssertionError("expected delete failure")


def test_first_upsert_skips_stale_delete_when_collection_does_not_exist() -> None:
    store = FakeVectorStore(fail_delete=True, collection_exists=False)
    index = MemoryVectorIndex(store)

    index.upsert(approved_memory())

    assert store.deleted == []
    assert len(store.added) == 1


def test_existing_collection_still_surfaces_delete_failure() -> None:
    store = FakeVectorStore(fail_delete=True, collection_exists=True)
    index = MemoryVectorIndex(store)

    with pytest.raises(RuntimeError, match="delete failed"):
        index.upsert(approved_memory())


def test_rebuild_index_clears_customer_and_indexes_records() -> None:
    namespace = ("customer", "C001", "memories")
    vector = FakeVectorStore()
    store = FakeMemoryStore({(namespace, "preference:shoe_size"): approved_memory()})

    class Repository:
        def list_indexable_memories(self, *, customer_id=None, limit=None):
            assert customer_id == "C001"
            assert limit is None
            return list(store.records.values())

    store.repository = Repository()
    service = MemoryRetrievalService(store, vector_index=MemoryVectorIndex(vector))

    assert service.rebuild_index(customer_id="C001") == 1
    assert vector.deleted[0]["expr"] == 'customer_id == "C001"'
    assert vector.added
