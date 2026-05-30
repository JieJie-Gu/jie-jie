# 写入演示用客户、商品、订单和知识库数据。

import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sqlalchemy.engine import make_url

from smart_cs.config import Settings
from smart_cs.infrastructure.database import Database
from smart_cs.infrastructure.repositories import SqlRepository


def ensure_sqlite_parent_directory(database_url: str) -> None:
    url = make_url(database_url)
    if url.get_backend_name() != "sqlite" or url.database in {None, "", ":memory:"}:
        return
    Path(url.database).expanduser().parent.mkdir(parents=True, exist_ok=True)


def main() -> None:
    settings = Settings()
    ensure_sqlite_parent_directory(settings.database_url)
    repository = SqlRepository(Database(settings.database_url))
    repository.create_schema()
    repository.seed_demo_data()
    print("Seeded demo customer C001 and delivered order O1001.")


if __name__ == "__main__":
    main()
