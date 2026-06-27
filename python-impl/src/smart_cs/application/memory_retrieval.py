# 复用向量检索底座召回长期记忆，并在注入前完成 SQL 回表和安全投影。
from __future__ import annotations

from datetime import UTC, datetime
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

    def get_by_id(self, memory_id: str) -> Any | None: ...

    def search(self, namespace: tuple[str, str, str], query: str, limit: int) -> list[Any]: ...


class MemoryVectorStoreProtocol(Protocol):
    def add_documents(self, documents: list[Document], **kwargs: Any) -> list[str]: ...

    def delete(
        self,
        ids: list[str] | None = None,
        expr: str | None = None,
        **kwargs: Any,
    ) -> bool | None: ...

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
            self.delete(memory, ignore_not_found=True)

    def upsert(self, memory: dict[str, Any]) -> None:
        document = memory_index_document(memory)
        memory_id = str(document.metadata["memory_id"])
        if self._collection_exists() is not False:
            if not self.delete(memory, ignore_not_found=True):
                raise RuntimeError("Unable to delete stale memory vector before upsert")
        self.store.add_documents([document], ids=[memory_id])

    def _collection_exists(self) -> bool | None:
        checker = getattr(self.store, "collection_exists", None)
        if not callable(checker):
            return None
        try:
            return checker()
        except Exception:
            LOGGER.warning("Unable to inspect memory collection; using guarded delete", exc_info=True)
            return None

    def delete(self, memory: dict[str, Any], *, ignore_not_found: bool = False) -> bool:
        memory_id = str(memory.get("memory_id") or memory.get("id") or "")
        if not memory_id:
            return True
        try:
            result = self.store.delete(ids=[memory_id])
        except Exception as error:
            if ignore_not_found and _is_not_found_error(error):
                return True
            raise
        return result is not False

    def rebuild_from_records(self, records: list[Any]) -> int:
        count = 0
        for record in records:
            memory = memory_record_to_dict(record)
            if not is_indexable_memory(memory):
                continue
            self.upsert(memory)
            count += 1
        return count

    def clear_customer(self, customer_id: str | None = None) -> bool:
        expression = (
            f'customer_id == "{_escape_expr(customer_id)}"'
            if customer_id is not None
            else 'namespace != ""'
        )
        try:
            result = self.store.delete(expr=expression)
        except Exception as error:
            if _is_not_found_error(error):
                return True
            raise
        return result is not False

    def search(self, *, customer_id: str, query: str, limit: int) -> list[Document]:
        namespace = f"customer/{customer_id}/memories"
        expression = (
            f'customer_id == "{_escape_expr(customer_id)}" '
            f'and namespace == "{_escape_expr(namespace)}" '
            'and review_status == "approved" '
            'and risk_level != "high" '
            'and confidence != "low" '
            'and memory_type != "sensitive_label" '
            'and memory_type != "risk_event" '
            'and memory_type != "badcase_candidate"'
        )
        return self.store.similarity_search(
            query,
            k=limit,
            expr=expression,
            fetch_k=max(limit * 4, 50),
            ranker_type="rrf",
            ranker_params={"k": 60},
        )


class MemoryRetrievalService:
    """Searches active customer memories through vector recall plus SQL fallback."""

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
        vector_records = self._vector_records(customer_id, query, limit)
        sql_records = self._sql_records(customer_id, query, limit)
        records = self._merge_records(vector_records, sql_records)
        return self._select(records, query=query, intent=intent, limit=limit, max_chars=max_chars)

    def rebuild_index(self, customer_id: str | None = None) -> int:
        if self.vector_index is None:
            return 0
        repository = getattr(self.memory_store, "repository", None)
        list_records = getattr(repository, "list_indexable_memories", None)
        if list_records is None:
            return 0
        self.vector_index.clear_customer(customer_id)
        return self.vector_index.rebuild_from_records(list_records(customer_id=customer_id))

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
            LOGGER.warning("Memory vector search failed; falling back to SQL", exc_info=True)
            return []
        for document in documents:
            memory_id = str(document.metadata.get("memory_id") or "")
            key = str(document.metadata.get("key") or "")
            marker = memory_id or f"{namespace}:{key}"
            if not marker or marker in seen:
                continue
            seen.add(marker)
            record = self._hydrate_vector_hit(
                namespace=namespace,
                memory_id=memory_id,
                key=key,
            )
            if record is not None and self._allowed_hydrated_record(record, customer_id):
                records.append(record)
        return records

    def _hydrate_vector_hit(
        self,
        *,
        namespace: tuple[str, str, str],
        memory_id: str,
        key: str,
    ) -> Any | None:
        if memory_id:
            getter_by_id = getattr(self.memory_store, "get_by_id", None)
            if getter_by_id is None:
                return None
            return getter_by_id(memory_id)
        if not key:
            return None
        getter = getattr(self.memory_store, "get", None)
        if getter is None:
            return None
        return getter(namespace, key)

    @staticmethod
    def _allowed_hydrated_record(record: Any, customer_id: str) -> bool:
        memory = memory_record_to_dict(record)
        return (
            memory.get("namespace") == f"customer/{customer_id}/memories"
            and memory.get("owner_id") == customer_id
            and is_indexable_memory(memory)
        )

    def _sql_records(self, customer_id: str, query: str, limit: int) -> list[Any]:
        namespace = ("customer", customer_id, "memories")
        search = getattr(self.memory_store, "search", None)
        if search is None:
            return []
        return search(namespace, query=query, limit=max(limit * 4, 20))

    @staticmethod
    def _merge_records(*record_groups: list[Any]) -> list[Any]:
        merged: list[Any] = []
        seen: set[str] = set()
        for group in record_groups:
            for record in group:
                memory = memory_record_to_dict(record)
                marker = str(memory.get("memory_id") or memory.get("id") or "")
                if not marker:
                    marker = f"{memory.get('namespace') or ''}:{memory.get('key') or ''}"
                if not marker or marker in seen:
                    continue
                seen.add(marker)
                merged.append(record)
        return merged


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
            compact_memory_value_for_index(memory),
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
            "risk_level": str(memory.get("risk_level") or ""),
            "confidence": str(memory.get("confidence") or ""),
            "expires_at": str(memory.get("expires_at") or ""),
            "updated_at": str(memory.get("updated_at") or ""),
        },
    )


def compact_memory_value_for_index(memory: dict[str, Any]) -> str:
    value = memory.get("value")
    if not isinstance(value, dict) or _has_forbidden_index_payload(value):
        return ""
    memory_kind = str(memory.get("memory_kind") or "")
    memory_type = str(memory.get("memory_type") or "")
    if memory_kind == "episodic":
        return _compact_allowed_pairs(
            value,
            {"order_id", "action_type", "status", "issue", "ticket_id"},
        )
    if memory_type == "preference":
        if {"attribute", "value", "unit"} & set(value):
            return _compact_allowed_pairs(value, {"attribute", "value", "unit"})
        return _compact_scalar_attributes(value)
    if memory_type == "profile":
        if {"field", "value", "unit"} & set(value):
            return _compact_allowed_pairs(value, {"field", "value", "unit"})
        return _compact_scalar_attributes(value)
    if memory_type == "constraint":
        if {"constraint_type", "constraint_value", "unit"} & set(value):
            return _compact_allowed_pairs(value, {"constraint_type", "constraint_value", "unit"})
        return _compact_scalar_attributes(value)
    return ""


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


def _has_forbidden_index_payload(value: dict[str, Any]) -> bool:
    forbidden = {
        "evidence",
        "before_json",
        "after_json",
        "business_result",
        "raw_tool_result",
        "memory_decision",
        "review_payload",
        "proposed_value",
        "previous_value",
        "conflict_with",
    }
    return bool(value.get("conflict")) or any(key in value for key in forbidden)


def _compact_allowed_pairs(value: dict[str, Any], allowed: set[str]) -> str:
    parts = []
    for key in sorted(allowed):
        item = value.get(key)
        if _is_scalar(item):
            parts.append(f"{key}: {item}")
    return "; ".join(parts)


def _compact_scalar_attributes(value: dict[str, Any]) -> str:
    parts = []
    for key, item in value.items():
        if _is_scalar(item):
            parts.append(f"attribute: {key}; value: {item}")
    return "; ".join(parts[:5])


def _is_scalar(value: Any) -> bool:
    return isinstance(value, (str, int, float, bool))


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


def _is_not_found_error(error: Exception) -> bool:
    text = str(error).lower()
    return "not found" in text or "not exist" in text or "does not exist" in text
