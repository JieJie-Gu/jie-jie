# 创建 SQLAlchemy engine、session 和数据库 schema。

from collections.abc import Iterator
from contextlib import contextmanager

from sqlalchemy import Engine, create_engine, event
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool


def _enable_sqlite_foreign_keys(dbapi_connection, _connection_record) -> None:
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.close()


class Database:
    """SQLAlchemy session management for a configured SQLite database URL."""

    def __init__(self, database_url: str) -> None:
        engine_options: dict = {}
        if database_url.startswith("sqlite"):
            engine_options["connect_args"] = {"check_same_thread": False}
            if database_url in {"sqlite://", "sqlite:///:memory:"}:
                engine_options["poolclass"] = StaticPool

        self.engine: Engine = create_engine(database_url, **engine_options)
        if self.engine.dialect.name == "sqlite":
            event.listen(self.engine, "connect", _enable_sqlite_foreign_keys)
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
