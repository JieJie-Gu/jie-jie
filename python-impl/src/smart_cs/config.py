from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="SMART_CS_", env_file=".env")

    database_url: str = "sqlite:///./data/smart_cs.db"
    checkpoint_path: Path = Path("data/checkpoints.db")
    model_mode: str = "llm"
    llm_model: str = "gpt-5.5"
    llm_base_url: str | None = "http://127.0.0.1:8317/v1"
    llm_api_key: str | None = "fab71afaca14f54043694ec31f0f70547b9ab98fe2363f760bbd8e0604268c3a"
    milvus_uri: str = "http://localhost:19530"
    milvus_collection: str = "smart_cs_knowledge"
    embedding_model: str = "BAAI/bge-m3"
    rag_enabled: bool = True
    asset_root: Path = Path("data/assets")
