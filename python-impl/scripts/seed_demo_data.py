import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from smart_cs.config import Settings
from smart_cs.infrastructure.database import Database
from smart_cs.infrastructure.repositories import SqlRepository


def main() -> None:
    settings = Settings()
    repository = SqlRepository(Database(settings.database_url))
    repository.create_schema()
    repository.seed_demo_data()
    print("Seeded demo customer C001 and delivered order O1001.")


if __name__ == "__main__":
    main()
