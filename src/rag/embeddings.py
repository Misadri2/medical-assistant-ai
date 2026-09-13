"""
Backends de embedding para o RAG.

- HashEmbeddings: determinístico, offline, zero dependências pesadas.
  Usado por padrão em dev/teste (inclusive nesta atividade, para o
  pipeline rodar sem GPU/internet). Não tem qualidade semântica real —
  serve para validar o encanamento do RAG (indexação, busca, citação de
  fonte), não para avaliar qualidade de recuperação.
- SentenceTransformerEmbeddings: embeddings reais (all-MiniLM-L6-v2),
  recomendado para produção. Requer `pip install sentence-transformers`
  (já listado no requirements.txt) e download do modelo na primeira
  execução.
"""
from __future__ import annotations

import hashlib

import numpy as np
from langchain_core.embeddings import Embeddings

from src.config import settings

_DIM = 384  # mesma dimensão do all-MiniLM-L6-v2, para troca transparente de backend


class HashEmbeddings(Embeddings):
    """Embedding determinístico baseado em hash de tokens. Só para dev/teste."""

    def _embed(self, text: str) -> list[float]:
        vec = np.zeros(_DIM, dtype=np.float32)
        for token in text.lower().split():
            h = int(hashlib.sha256(token.encode("utf-8")).hexdigest(), 16)
            vec[h % _DIM] += 1.0
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec.tolist()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


def get_embeddings() -> Embeddings:
    if settings.embedding_backend == "sentence-transformers":
        from langchain_community.embeddings import HuggingFaceEmbeddings

        return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return HashEmbeddings()
