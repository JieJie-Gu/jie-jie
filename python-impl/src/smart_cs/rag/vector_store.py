from __future__ import annotations

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_milvus import BM25BuiltInFunction, Milvus

from smart_cs.config import Settings


def build_hybrid_store(
    settings: Settings,
    embeddings: Embeddings,
    documents: list[Document],
    *,
    drop_old: bool,
) -> Milvus:
    """Build the official Milvus dense plus BM25 hybrid knowledge store."""

    return Milvus.from_documents(
        documents=documents,
        embedding=embeddings,
        builtin_function=BM25BuiltInFunction(analyzer_params={"type": "chinese"}),
        vector_field=["dense", "sparse"],
        connection_args={"uri": settings.milvus_uri},
        collection_name=settings.milvus_collection,
        consistency_level="Strong",
        drop_old=drop_old,
    )
