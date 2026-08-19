"""Application configuration loaded from environment variables / .env file."""

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Central settings for the arhivadoc.eu OCR backend."""

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # --- OCR ---
    tesseract_lang: str = "ron"            # primary Tesseract language
    tesseract_fallback_lang: str = "ron+eng"  # fallback combination
    ocr_dpi: int = 300                     # rasterization DPI for PDF pages
    ocr_low_conf_threshold: float = 60.0   # words below this confidence get flagged

    # --- LLM providers ---
    llm_provider: str = "ollama"           # ollama | openai | none
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "qwen2.5:7b"
    openai_base_url: str = "https://api.openai.com/v1"
    openai_api_key: str = ""
    openai_model: str = "gpt-4o-mini"
    llm_timeout_seconds: float = 120.0
    llm_max_block_chars: int = 3500        # max chars per correction request

    # --- Storage / ingest ---
    archive_root: str = "./archive"        # final archive tree root
    watch_folder: str = "./watch"          # scanner output folder to watch
    data_dir: str = "./data"               # jobs / temp working directory

    # --- Server ---
    host: str = "0.0.0.0"
    port: int = 8000

    def ensure_dirs(self) -> None:
        """Create required runtime directories if they do not exist."""
        for p in (self.archive_root, self.watch_folder, self.data_dir):
            Path(p).mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    return Settings()
