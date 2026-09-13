"""
Wrapper de recuperação de contexto (RAG) que devolve, junto com o texto
recuperado, a fonte (protocolo) de onde veio — requisito de explainability
do desafio ("indicar a fonte da informação utilizada na resposta").
"""
from __future__ import annotations

from dataclasses import dataclass

from langchain_community.vectorstores import Chroma


@dataclass
class RetrievedChunk:
    conteudo: str
    fonte_id: str
    titulo: str
    fonte_documento: str
    score: float | None = None


class ProtocolRetriever:
    def __init__(self, vectorstore: Chroma, k: int = 3):
        self._vectorstore = vectorstore
        self._k = k

    def retrieve(self, query: str) -> list[RetrievedChunk]:
        results = self._vectorstore.similarity_search_with_score(query, k=self._k)
        chunks = []
        for doc, score in results:
            chunks.append(
                RetrievedChunk(
                    conteudo=doc.page_content,
                    fonte_id=doc.metadata.get("fonte_id", "desconhecida"),
                    titulo=doc.metadata.get("titulo", ""),
                    fonte_documento=doc.metadata.get("fonte_documento", ""),
                    score=score,
                )
            )
        return chunks

    def format_context_with_sources(self, chunks: list[RetrievedChunk]) -> str:
        """Formata o contexto recuperado para o prompt, com marcação de fonte."""
        partes = []
        for c in chunks:
            partes.append(f"[Fonte: {c.fonte_id} — {c.titulo}]\n{c.conteudo}")
        return "\n\n".join(partes)
