from __future__ import annotations

import pytest

from src.config import settings


@pytest.fixture(autouse=True)
def _isolated_settings(tmp_path, monkeypatch):
    """
    Garante que cada teste rode com banco, vector store e log de auditoria
    isolados em um diretório temporário, evitando efeitos colaterais entre
    testes ou com dados gerados pela demo (`python -m src.main`).

    Muta os atributos do singleton `settings` já importado (em vez de
    substituir o objeto), já que outros módulos fazem
    `from src.config import settings` e guardam a própria referência ao
    mesmo objeto — só assim a alteração é enxergada por todos eles.
    """
    monkeypatch.setattr(settings, "database_url", f"sqlite:///{tmp_path / 'test.db'}")
    monkeypatch.setattr(settings, "chroma_persist_dir", str(tmp_path / "chroma_db"))
    monkeypatch.setattr(settings, "audit_log_path", str(tmp_path / "audit_log.jsonl"))
    monkeypatch.setattr(settings, "embedding_backend", "hash")
    monkeypatch.setattr(settings, "llm_backend", "mock")
    monkeypatch.setattr(settings, "anonymizer_backend", "regex")
    yield
