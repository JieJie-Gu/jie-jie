from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="SMART_CS_", env_file=".env")

    database_url: str = "sqlite:///./data/smart_cs.db"
    checkpoint_path: Path = Path("data/checkpoints.db")
    model_mode: str = "rules"
    llm_model: str = "gpt-4o-mini"
    llm_base_url: str | None = None
    llm_api_key: str | None = None
    milvus_uri: str = "http://localhost:19530"
    milvus_collection: str = "smart_cs_knowledge"
    embedding_model: str = "BAAI/bge-small-zh-v1.5"
    rag_enabled: bool = False
