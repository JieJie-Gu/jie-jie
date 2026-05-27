from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from smart_cs.config import Settings
from smart_cs.rag.embeddings import LocalSentenceEmbeddings
from smart_cs.rag.indexing import load_knowledge_documents
from smart_cs.rag.vector_store import build_hybrid_store


def main() -> None:
    settings = Settings()
    documents = load_knowledge_documents(ROOT / "data" / "knowledge")
    embeddings = LocalSentenceEmbeddings(settings.embedding_model)
    build_hybrid_store(settings, embeddings, documents, drop_old=True)
    print(f"Indexed {len(documents)} knowledge sentence windows into {settings.milvus_collection}.")


if __name__ == "__main__":
    main()
