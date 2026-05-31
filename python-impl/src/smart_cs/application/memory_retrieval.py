# 复用向量检索底座召回长期记忆，并在注入前完成 SQL 回表和安全投影。
from __future__ import annotations

from datetime import UTC, datetime
import json
import logging
from typing import Any, Protocol

from langchain_core.documents import Document

from smart_cs.application.memory_selector import (
    MemoryContextSelector,
    MemorySelectionInput,
)


LOGGER = logging.getLogger(__name__)


class MemoryStoreProtocol(Protocol):
    def get(self, namespace: tuple[str, str, str], key: str) -> Any | None: ...

    def search(self, namespace: tuple[str, str, str], query: str, limit: int) -> list[Any]: ...


class MemoryVectorStoreProtocol(Protocol):
    def add_documents(self, documents: list[Document], **kwargs: Any) -> list[str]: ...

    def delete(self, ids: list[str] | None = None, expr: str | None = None, **kwargs: Any) -> bool | None: ...

    def similarity_search(self, query: str, **kwargs: Any) -> list[Document]: ...


class MemoryVectorIndex:
    """Maintains a searchable Milvus copy of approved customer memories."""

    def __init__(self, store: MemoryVectorStoreProtocol) -> None:
        self.store = store

    def sync_record(self, record: Any) -> None:
        memory = memory_record_to_dict(record)
        if is_indexable_memory(memory):
            self.upsert(memory)
        else:
            self.delete(memory)

    def upsert(self, memory: dict[str, Any]) -> None:
        document = memory_index_document(memory)
        memory_id = str(document.metadata["memory_id"])
        self.delete(memory)
        self.store.add_documents([document], ids=[memory_id])

    def delete(self, memory: dict[str, Any]) -> None:
        memory_id = str(memory.get("memory_id") or memory.get("id") or "")
        if not memory_id:
            return
        try:
            self.store.delete(ids=[memory_id])
        except Exception:
            LOGGER.debug("Unable to delete memory vector by id", exc_info=True)

    def search(self, *, customer_id: str, query: str, limit: int) -> list[Document]:
        namespace = f"customer/{customer_id}/memories"
        expression = (
            f'customer_id == "{_escape_expr(customer_id)}" '
            f'and namespace == "{_escape_expr(namespace)}" '
            'and review_status == "approved"'
        )
        return self.store.similarity_search(
            query,
            k=limit,
            expr=expression,
            fetch_k=max(limit * 2, 20),
            ranker_type="rrf",
            ranker_params={"k": 60},
        )


class MemoryRetrievalService:
    """Searches active customer memories through vector recall with SQL fallback."""

    def __init__(
        self,
        memory_store: MemoryStoreProtocol,
        *,
        selector: MemoryContextSelector | None = None,
        vector_index: MemoryVectorIndex | None = None,
    ) -> None:
        self.memory_store = memory_store
        self.selector = selector or MemoryContextSelector()
        self.vector_index = vector_index

    def search_active_memories(
        self,
        *,
        customer_id: str,
        query: str,
        intent: str | None = None,
        limit: int = 5,
        max_chars: int = 1200,
    ) -> list[dict[str, Any]]:
        used_vector = self.vector_index is not None
        records = self._vector_records(customer_id, query, limit)
        if not records:
            records = self._sql_records(customer_id, query, limit)
            used_vector = False
        selected = self._select(records, query=query, intent=intent, limit=limit, max_chars=max_chars)
        if not selected and used_vector:
            records = self._sql_records(customer_id, query, limit)
            selected = self._select(records, query=query, intent=intent, limit=limit, max_chars=max_chars)
        return selected

    def _select(
        self,
        records: list[Any],
        *,
        query: str,
        intent: str | None,
        limit: int,
        max_chars: int,
    ) -> list[dict[str, Any]]:
        raw_memories = [memory_record_to_dict(record) for record in records]
        selected = self.selector.select(
            MemorySelectionInput(
                query=query,
                intent=intent,
                memories=raw_memories,
                limit=limit,
                max_chars=max_chars,
            )
        )
        return [memory.model_dump() for memory in selected.memories]

    def _vector_records(self, customer_id: str, query: str, limit: int) -> list[Any]:
        if self.vector_index is None:
            return []
        namespace = ("customer", customer_id, "memories")
        records: list[Any] = []
        seen: set[str] = set()
        try:
            documents = self.vector_index.search(
                customer_id=customer_id,
                query=query,
                limit=max(limit * 4, 20),
            )
        except Exception:
            LOGGER.debug("Memory vector search failed; falling back to SQL", exc_info=True)
            return []
        for document in documents:
            key = str(document.metadata.get("key") or "")
            if not key or key in seen:
                continue
            seen.add(key)
            getter = getattr(self.memory_store, "get", None)
            if getter is None:
                continue
            record = getter(namespace, key)
            if record is not None:
                records.append(record)
        return records

    def _sql_records(self, customer_id: str, query: str, limit: int) -> list[Any]:
        namespace = ("customer", customer_id, "memories")
        search = getattr(self.memory_store, "search", None)
        if search is None:
            return []
        return search(namespace, query=query, limit=max(limit * 4, 20))


def memory_index_document(memory: dict[str, Any]) -> Document:
    memory_id = str(memory.get("memory_id") or memory.get("id") or "")
    key = str(memory.get("key") or memory_id.split(":", 1)[-1])
    namespace = str(memory.get("namespace") or "")
    customer_id = str(memory.get("owner_id") or memory.get("customer_id") or "")
    content = "\n".join(
        part
        for part in [
            str(memory.get("title") or ""),
            str(memory.get("description") or ""),
            str(memory.get("memory_kind") or ""),
            str(memory.get("memory_type") or ""),
            json.dumps(memory.get("value") or {}, ensure_ascii=False, default=str),
        ]
        if part
    )
    return Document(
        page_content=content,
        metadata={
            "memory_id": memory_id,
            "key": key,
            "namespace": namespace,
            "customer_id": customer_id,
            "memory_kind": str(memory.get("memory_kind") or ""),
            "memory_type": str(memory.get("memory_type") or ""),
            "review_status": str(memory.get("review_status") or ""),
            "expires_at": str(memory.get("expires_at") or ""),
            "updated_at": str(memory.get("updated_at") or ""),
        },
    )


def is_indexable_memory(memory: dict[str, Any]) -> bool:
    namespace = str(memory.get("namespace") or "")
    if not namespace.startswith("customer/") or not namespace.endswith("/memories"):
        return False
    if memory.get("review_status") != "approved":
        return False
    if memory.get("risk_level") == "high" or memory.get("confidence") == "low":
        return False
    if str(memory.get("memory_type") or "") in {
        "sensitive_label",
        "risk_event",
        "badcase_candidate",
    }:
        return False
    return not _is_expired(memory.get("expires_at"))


def memory_record_to_dict(record: Any) -> dict[str, Any]:
    if isinstance(record, dict):
        memory = dict(record)
    else:
        value = getattr(record, "value", None)
        if isinstance(value, dict):
            memory = dict(value)
        else:
            value_json = getattr(record, "value_json", None)
            memory = dict(value_json) if isinstance(value_json, dict) else {}
        memory.setdefault("memory_id", getattr(record, "id", None))
        memory.setdefault("id", getattr(record, "id", None))
        memory.setdefault("namespace", getattr(record, "namespace", None))
        memory.setdefault("key", getattr(record, "key", None))
        memory.setdefault("title", getattr(record, "title", None))
        memory.setdefault("description", getattr(record, "description", None))
        memory.setdefault("memory_type", getattr(record, "memory_type", None))
        memory.setdefault("scope", getattr(record, "scope", None))
        memory.setdefault("owner_id", getattr(record, "owner_id", None))
        memory.setdefault("confidence", getattr(record, "confidence", None))
        memory.setdefault("risk_level", getattr(record, "risk_level", None))
        memory.setdefault("review_status", getattr(record, "review_status", None))
        _set_datetime(memory, "created_at", getattr(record, "created_at", None))
        _set_datetime(memory, "updated_at", getattr(record, "updated_at", None))
        _set_datetime(memory, "expires_at", getattr(record, "expires_at", None))
    if "memory_id" not in memory and memory.get("namespace") and memory.get("key"):
        memory["memory_id"] = f"{memory['namespace']}:{memory['key']}"
    return {key: value for key, value in memory.items() if value is not None}


def _set_datetime(memory: dict[str, Any], key: str, value: Any) -> None:
    if value is None or memory.get(key):
        return
    memory[key] = value.isoformat() if hasattr(value, "isoformat") else str(value)


def _is_expired(value: Any) -> bool:
    if not value:
        return False
    try:
        expires_at = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return False
    if expires_at.tzinfo is None:
        expires_at = expires_at.replace(tzinfo=UTC)
    return expires_at <= datetime.now(UTC)


def _escape_expr(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"')
