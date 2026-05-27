from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from smart_cs.config import Settings
from smart_cs.rag.embeddings import LocalSentenceEmbeddings
from smart_cs.rag.indexing import markdown_sentence_documents
from smart_cs.rag.vector_store import build_hybrid_store


CATEGORIES = {
    "after_sales_policy": "after_sales",
    "shipping_policy": "shipping",
    "product_guide": "product",
    "faq": "faq",
}


def load_knowledge_documents() -> list:
    documents = []
    for document_id, category in CATEGORIES.items():
        markdown = (ROOT / "data" / "knowledge" / f"{document_id}.md").read_text(
            encoding="utf-8"
        )
        documents.extend(markdown_sentence_documents(document_id, category, markdown))
    return documents


def main() -> None:
    settings = Settings()
    documents = load_knowledge_documents()
    embeddings = LocalSentenceEmbeddings(settings.embedding_model)
    build_hybrid_store(settings, embeddings, documents, drop_old=True)
    print(f"Indexed {len(documents)} knowledge sentence windows into {settings.milvus_collection}.")


if __name__ == "__main__":
    main()
