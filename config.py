"""Настройки приложения и список источников НПА."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import yaml

ROOT = Path(__file__).parent
VALID_PROVIDERS = {"ollama", "anthropic", "openai"}


@dataclass(frozen=True)
class Settings:
    llm_provider: str
    ollama_url: str
    gen_model: str
    embed_model: str
    num_ctx: int
    embed_dim: int
    relevance_threshold: float
    db_path: Path
    raw_dir: Path
    upload_dir: Path
    max_upload_mb: int
    max_clauses: int
    max_chunk_chars: int = 2000
    min_chunk_chars: int = 100


@dataclass(frozen=True)
class Source:
    key: str
    title: str
    url: str
    doc_type: str


def _load_dotenv(env_path: Path | None = None) -> dict[str, str]:
    """Парсит .env в словарь, не трогая os.environ."""
    env_file = env_path or (ROOT / ".env")
    if not env_file.exists():
        return {}
    values: dict[str, str] = {}
    for line in env_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        values[key.strip()] = value.strip()
    return values


def load_settings(env_path: Path | None = None) -> Settings:
    # .env — только дефолты; реальные переменные окружения имеют приоритет.
    env = {**_load_dotenv(env_path), **os.environ}
    provider = env.get("LLM_PROVIDER", "ollama")
    if provider not in VALID_PROVIDERS:
        raise ValueError(
            f"LLM_PROVIDER={provider!r} не поддерживается. "
            f"Допустимые значения: {', '.join(sorted(VALID_PROVIDERS))}"
        )
    return Settings(
        llm_provider=provider,
        ollama_url=env.get("OLLAMA_URL", "http://localhost:11434"),
        gen_model=env.get("GEN_MODEL", "qwen3:4b-instruct"),
        embed_model=env.get("EMBED_MODEL", "bge-m3"),
        num_ctx=int(env.get("NUM_CTX", "6144")),
        embed_dim=int(env.get("EMBED_DIM", "1024")),
        relevance_threshold=float(env.get("RELEVANCE_THRESHOLD", "0.45")),
        db_path=ROOT / env.get("DB_PATH", "law.db"),
        raw_dir=ROOT / "data" / "raw",
        upload_dir=ROOT / "data" / "uploads",
        max_upload_mb=int(env.get("MAX_UPLOAD_MB", "10")),
        max_clauses=int(env.get("MAX_CLAUSES", "300")),
    )


def load_sources(path: Path | None = None) -> list[Source]:
    path = path or ROOT / "config" / "sources.yaml"
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    sources: list[Source] = []
    seen: set[str] = set()
    for item in data["sources"]:
        if item["key"] in seen:
            raise ValueError(f"В {path.name} дублирующийся ключ: {item['key']}")
        seen.add(item["key"])
        sources.append(
            Source(
                key=item["key"],
                title=item["title"],
                url=item["url"],
                doc_type=item["doc_type"],
            )
        )
    return sources
