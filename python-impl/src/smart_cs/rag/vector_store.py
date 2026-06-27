# 封装 Milvus 集合管理、写入和混合检索。

from __future__ import annotations

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_milvus import BM25BuiltInFunction, Milvus
from pymilvus import connections

from smart_cs.config import Settings


DENSE_VECTOR_FIELD = "dense"
SPARSE_VECTOR_FIELD = "sparse"
VECTOR_FIELDS = [DENSE_VECTOR_FIELD, SPARSE_VECTOR_FIELD]


def hybrid_index_params() -> list[dict]:
    return [
        {
            "metric_type": "COSINE",
            "index_type": "HNSW",
            "params": {"M": 16, "efConstruction": 200},
        },
        {
            "metric_type": "BM25",
            "index_type": "AUTOINDEX",
            "params": {},
        },
    ]


def hybrid_search_params() -> list[dict]:
    return [
        {"metric_type": "COSINE", "params": {"ef": 64}},
        {"metric_type": "BM25", "params": {}},
    ]


def hybrid_store_kwargs(settings: Settings, *, collection_name: str) -> dict:
    return {
        "builtin_function": BM25BuiltInFunction(analyzer_params={"type": "chinese"}),
        "vector_field": VECTOR_FIELDS,
        "connection_args": {
            "uri": settings.milvus_uri,
            "timeout": settings.milvus_timeout_seconds,
        },
        "collection_name": collection_name,
        "consistency_level": "Strong",
        "index_params": hybrid_index_params(),
        "search_params": hybrid_search_params(),
    }


class _OrmAliasMilvus(Milvus):
    """Bridge MilvusClient aliases to PyMilvus ORM calls used by langchain-milvus."""

    @property
    def col(self):
        self._ensure_orm_alias()
        return super().col

    @col.setter
    def col(self, value) -> None:
        Milvus.col.fset(self, value)

    def _ensure_orm_alias(self) -> None:
        if not getattr(self, "alias", None):
            return
        if connections.has_connection(self.alias):
            return
        connections.connect(alias=self.alias, **self._connection_args)

    def collection_exists(self) -> bool:
        """Return whether the configured collection already exists."""

        return bool(self.client.has_collection(collection_name=self.collection_name))


def build_hybrid_store(
    settings: Settings,
    embeddings: Embeddings,
    documents: list[Document],
    *,
    drop_old: bool,
    collection_name: str | None = None,
) -> Milvus:
    """Build the official Milvus dense plus BM25 hybrid knowledge store."""

    return _OrmAliasMilvus.from_documents(
        documents=documents,
        embedding=embeddings,
        **hybrid_store_kwargs(
            settings,
            collection_name=collection_name or settings.milvus_collection,
        ),
        drop_old=drop_old,
    )


def connect_hybrid_store(
    settings: Settings,
    embeddings: Embeddings,
    *,
    collection_name: str | None = None,
) -> Milvus:
    """Connect retrieval to an indexed collection without rebuilding it."""

    return _OrmAliasMilvus(
        embedding_function=embeddings,
        **hybrid_store_kwargs(
            settings,
            collection_name=collection_name or settings.milvus_collection,
        ),
    )
