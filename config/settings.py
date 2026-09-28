"""
AFRA Settings — Centralized configuration management.

Uses Pydantic BaseSettings to load configuration from environment variables
and .env files. All sensitive values (API keys) are loaded from environment
only and never hardcoded.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables and .env file.

    Priority order (highest to lowest):
    1. Environment variables
    2. .env file
    3. Default values defined here
    """

    # ── LLM Provider ──────────────────────────────────────────────────────
    llm_provider: Literal["openai", "anthropic"] = "openai"
    openai_api_key: str = ""
    anthropic_api_key: str = ""
    llm_model: str = "gpt-4o-mini"
    embedding_model: str = "text-embedding-3-small"

    # ── Financial Data APIs ───────────────────────────────────────────────
    fmp_api_key: str = ""
    alpha_vantage_api_key: str = ""

    # ── Search & News APIs ────────────────────────────────────────────────
    tavily_api_key: str = ""
    newsapi_key: str = ""

    # ── SEC EDGAR ─────────────────────────────────────────────────────────
    sec_edgar_user_agent: str = "AFRA research-agent contact@example.com"

    # ── Vector Database ───────────────────────────────────────────────────
    vector_db: Literal["chroma", "pinecone", "qdrant"] = "chroma"
    chroma_persist_dir: str = "./data/chroma"
    pinecone_api_key: str = ""
    pinecone_index_name: str = "afra-research"

    # ── Agent Configuration ───────────────────────────────────────────────
    max_tool_calls: int = 20
    max_retries: int = 5

    # ── Logging ───────────────────────────────────────────────────────────
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"

    # ── Paths ─────────────────────────────────────────────────────────────
    project_root: Path = Path(__file__).resolve().parent.parent

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore",
    }

    # ── Derived properties ────────────────────────────────────────────────

    @property
    def active_api_key(self) -> str:
        """Return the API key for the active LLM provider."""
        if self.llm_provider == "openai":
            return self.openai_api_key
        return self.anthropic_api_key

    @property
    def data_dir(self) -> Path:
        """Return the data directory path, creating it if needed."""
        path = self.project_root / "data"
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def chroma_path(self) -> Path:
        """Return the Chroma persistent storage path."""
        path = Path(self.chroma_persist_dir)
        if not path.is_absolute():
            path = self.project_root / path
        path.mkdir(parents=True, exist_ok=True)
        return path


@lru_cache
def get_settings() -> Settings:
    """Return a cached singleton Settings instance.

    Using lru_cache ensures the .env file is read only once and the same
    Settings object is reused across the application.
    """
    return Settings()
