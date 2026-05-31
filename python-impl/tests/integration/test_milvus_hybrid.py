# 测试 Milvus 混合检索集成链路。

from __future__ import annotations

import hashlib
import socket
from urllib.parse import urlparse

import pytest
from langchain_core.embeddings import Embeddings

from smart_cs.config import Settings
from smart_cs.rag.indexing import markdown_section_documents
from smart_cs.rag.vector_store import build_hybrid_store


class DeterministicEmbeddings(Embeddings):
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_query(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        return [byte / 255 for byte in digest[:16]]


def require_milvus(uri: str) -> None:
    endpoint = urlparse(uri)
    try:
        with socket.create_connection((endpoint.hostname or "localhost", endpoint.port or 19530), 0.2):
            pass
    except OSError:
        pytest.skip(f"Milvus is not reachable at {uri}; start docker-compose standalone.")


@pytest.mark.integration
def test_hybrid_search_returns_filtered_after_sales_section() -> None:
    settings = Settings(milvus_collection="smart_cs_knowledge_test")
    require_milvus(settings.milvus_uri)
    documents = markdown_section_documents(
        "after_sales_policy",
        "after_sales",
        "# 售后政策\n## 七天无理由\n签收后七天内可以申请退货。商品应保持完好。",
    )
    documents += markdown_section_documents(
        "shipping_policy",
        "shipping",
        "# 配送说明\n## 发货状态\n已发货表示包裹已经交给承运方处理。",
    )
    store = build_hybrid_store(settings, DeterministicEmbeddings(), documents, drop_old=True)

    results = store.similarity_search(
        "鞋子退货期限",
        k=2,
        expr='category == "after_sales"',
        ranker_type="rrf",
        ranker_params={"k": 60},
    )

    assert results
    assert all(item.metadata["category"] == "after_sales" for item in results)
