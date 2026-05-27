from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="AIE_RAG_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    artifacts_dir: Path = Path("artifacts")
    kb_path: Path = Path("data/knowledge_base.json")

    embedding_model: str = "paraphrase-multilingual-MiniLM-L12-v2"
    chunk_size: int = 320
    chunk_overlap: int = 60

    log_level: str = "INFO"


def get_settings() -> Settings:
    return Settings()

