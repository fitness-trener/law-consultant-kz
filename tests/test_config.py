from pathlib import Path

import pytest

from config import Source, load_settings, load_sources


def test_load_settings_uses_defaults(monkeypatch):
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    monkeypatch.delenv("GEN_MODEL", raising=False)
    s = load_settings()
    assert s.llm_provider == "ollama"
    assert s.gen_model == "qwen3:4b-instruct"
    assert s.embed_model == "bge-m3"
    assert s.embed_dim == 1024
    assert s.num_ctx == 6144
    assert s.relevance_threshold == 0.45


def test_load_settings_reads_env(monkeypatch):
    monkeypatch.setenv("GEN_MODEL", "qwen2.5:3b-instruct")
    monkeypatch.setenv("RELEVANCE_THRESHOLD", "0.6")
    s = load_settings()
    assert s.gen_model == "qwen2.5:3b-instruct"
    assert s.relevance_threshold == 0.6


def test_load_settings_rejects_unknown_provider(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    with pytest.raises(ValueError, match="LLM_PROVIDER"):
        load_settings()


def test_load_sources_returns_acts():
    sources = load_sources()
    assert len(sources) >= 20
    labour = next(s for s in sources if s.key == "labour_code")
    assert labour.url.startswith("https://adilet.zan.kz/rus/docs/")
    assert labour.doc_type == "кодекс"
    assert all(isinstance(s, Source) for s in sources)


def test_load_sources_rejects_duplicate_keys(tmp_path):
    bad = tmp_path / "sources.yaml"
    bad.write_text(
        "sources:\n"
        "  - key: a\n    title: A\n    url: https://adilet.zan.kz/rus/docs/A\n    doc_type: закон\n"
        "  - key: a\n    title: B\n    url: https://adilet.zan.kz/rus/docs/B\n    doc_type: закон\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="дублирующийся ключ"):
        load_sources(bad)
