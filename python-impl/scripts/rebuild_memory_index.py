# 从 SQL 权威记忆表重建 Milvus memory 向量索引。
from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from smart_cs.application.memory_retrieval import MemoryVectorIndex  # noqa: E402
from smart_cs.config import Settings  # noqa: E402
from smart_cs.infrastructure.database import Database  # noqa: E402
from smart_cs.infrastructure.repositories import SqlRepository  # noqa: E402
from smart_cs.rag.embeddings import LocalSentenceEmbeddings  # noqa: E402
from smart_cs.rag.vector_store import connect_hybrid_store  # noqa: E402


def rebuild_memory_index(customer_id: str | None = None) -> int:
    settings = Settings()
    database = Database(settings.database_url)
    try:
        repository = SqlRepository(database)
        repository.create_schema()
        embeddings = LocalSentenceEmbeddings(settings.embedding_model)
        index = MemoryVectorIndex(
            connect_hybrid_store(
                settings,
                embeddings,
                collection_name=settings.memory_milvus_collection,
            )
        )
        index.clear_customer(customer_id)
        return index.rebuild_from_records(
            repository.list_indexable_memories(customer_id=customer_id)
        )
    finally:
        database.dispose()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--customer-id", default=None)
    args = parser.parse_args()

    try:
        count = rebuild_memory_index(customer_id=args.customer_id)
    except Exception as error:
        print(
            f"Unable to rebuild memory index: {type(error).__name__}: {error}",
            file=sys.stderr,
        )
        raise SystemExit(2) from error
    scope = args.customer_id or "all customers"
    print(f"Rebuilt {count} approved active memories for {scope}.")


if __name__ == "__main__":
    main()
