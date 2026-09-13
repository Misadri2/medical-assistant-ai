"""
Constrói e persiste a base vetorial (Chroma) a partir dos protocolos
hospitalares sintéticos, para consulta pelo assistente (RAG).

Cada chunk guarda metadados de fonte (id do protocolo, título) para
permitir explainability: toda resposta do assistente deve poder apontar
de qual protocolo veio a informação usada.
"""
from __future__ import annotations

import json
from pathlib import Path

from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

from src.config import settings
from src.rag.embeddings import get_embeddings

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "synthetic"


def load_protocol_documents() -> list[Document]:
    with open(DATA_DIR / "protocolos_hospitalares.json", encoding="utf-8") as f:
        protocolos = json.load(f)

    docs = []
    for p in protocolos:
        docs.append(
            Document(
                page_content=p["conteudo"],
                metadata={
                    "fonte_id": p["id"],
                    "titulo": p["titulo"],
                    "categoria": p["categoria"],
                    "fonte_documento": p["fonte"],
                },
            )
        )
    return docs


def build_vectorstore(persist: bool = True) -> Chroma:
    docs = load_protocol_documents()
    embeddings = get_embeddings()

    kwargs = {}
    if persist:
        kwargs["persist_directory"] = settings.chroma_persist_dir

    vectorstore = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        collection_name="protocolos_hospitalares",
        **kwargs,
    )
    return vectorstore


def load_vectorstore() -> Chroma:
    """Carrega a base já persistida em disco (usado em produção/demo)."""
    return Chroma(
        collection_name="protocolos_hospitalares",
        embedding_function=get_embeddings(),
        persist_directory=settings.chroma_persist_dir,
    )


if __name__ == "__main__":
    vs = build_vectorstore(persist=True)
    print(f"Vector store criada com {len(load_protocol_documents())} documentos.")
