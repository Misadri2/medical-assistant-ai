"""
Configuração central do projeto.

Lê variáveis de ambiente (via .env, se presente) e expõe um objeto `settings`
único, usado por todos os outros módulos. Isso evita `os.environ.get(...)`
espalhado pelo código e facilita trocar backends (mock <-> produção) sem
alterar lógica de negócio.
"""
from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # LLM
    llm_backend: str = "mock"  # "mock" | "ollama"
    ollama_model_name: str = "medassist-llama3-8b"
    ollama_host: str = "http://localhost:11434"

    # Hugging Face
    hf_token: str = ""
    hf_repo_id: str = ""

    # Banco de dados
    database_url: str = f"sqlite:///{BASE_DIR / 'data' / 'processed' / 'hospital.db'}"

    # RAG / Vector store
    chroma_persist_dir: str = str(BASE_DIR / "data" / "processed" / "chroma_db")
    embedding_backend: str = "hash"  # "hash" | "sentence-transformers"

    # Anonimização
    anonymizer_backend: str = "regex"  # "regex" | "presidio"

    # Auditoria
    audit_log_path: str = str(BASE_DIR / "data" / "processed" / "audit_log.jsonl")


settings = Settings()
