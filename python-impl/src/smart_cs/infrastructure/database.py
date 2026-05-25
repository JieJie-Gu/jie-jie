from collections.abc import Iterator
from contextlib import contextmanager

from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool


class Database:
    """SQLAlchemy session management for a configured SQLite database URL."""

    def __init__(self, database_url: str) -> None:
        engine_options: dict = {}
        if database_url.startswith("sqlite"):
            engine_options["connect_args"] = {"check_same_thread": False}
            if database_url in {"sqlite://", "sqlite:///:memory:"}:
                engine_options["poolclass"] = StaticPool

        self.engine: Engine = create_engine(database_url, **engine_options)
        self._sessions = sessionmaker(bind=self.engine, expire_on_commit=False)

    @contextmanager
    def session(self) -> Iterator[Session]:
        session = self._sessions()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def dispose(self) -> None:
        self.engine.dispose()
