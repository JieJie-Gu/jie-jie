from __future__ import annotations

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_milvus import BM25BuiltInFunction, Milvus
from pymilvus import connections

from smart_cs.config import Settings


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


def build_hybrid_store(
    settings: Settings,
    embeddings: Embeddings,
    documents: list[Document],
    *,
    drop_old: bool,
) -> Milvus:
    """Build the official Milvus dense plus BM25 hybrid knowledge store."""

    return _OrmAliasMilvus.from_documents(
        documents=documents,
        embedding=embeddings,
        builtin_function=BM25BuiltInFunction(analyzer_params={"type": "chinese"}),
        vector_field=["dense", "sparse"],
        connection_args={"uri": settings.milvus_uri},
        collection_name=settings.milvus_collection,
        consistency_level="Strong",
        drop_old=drop_old,
    )


def connect_hybrid_store(settings: Settings, embeddings: Embeddings) -> Milvus:
    """Connect retrieval to an indexed collection without rebuilding it."""

    return _OrmAliasMilvus(
        embedding_function=embeddings,
        builtin_function=BM25BuiltInFunction(analyzer_params={"type": "chinese"}),
        vector_field=["dense", "sparse"],
        connection_args={"uri": settings.milvus_uri},
        collection_name=settings.milvus_collection,
        consistency_level="Strong",
    )
