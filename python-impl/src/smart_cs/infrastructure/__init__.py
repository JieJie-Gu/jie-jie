"""Persistence adapters for the smart customer service application."""

from smart_cs.infrastructure.database import Database
from smart_cs.infrastructure.repositories import SqlRepository

__all__ = ["Database", "SqlRepository"]
