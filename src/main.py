"""
Ponto de entrada para rodar uma demonstração completa do assistente
médico: inicializa banco, vector store, monta o grafo e executa uma
pergunta de exemplo.

Uso:
    python -m src.main --paciente-id 1 --pergunta "Qual a conduta para lactato elevado?"
"""
from __future__ import annotations

import argparse

from src.audit.logger import get_default_logger
from src.database.init_db import init_db
from src.llm.inference import get_llm
from src.rag.retriever import ProtocolRetriever
from src.rag.vectorstore import build_vectorstore
from src.graph.workflow import build_workflow


def run_demo(paciente_id: int, pergunta: str) -> None:
    print("1) Inicializando banco de dados (SQLite)...")
    init_db(seed=True)

    print("2) Construindo vector store (Chroma) com protocolos hospitalares...")
    vectorstore = build_vectorstore(persist=True)
    retriever = ProtocolRetriever(vectorstore, k=2)

    print("3) Carregando LLM (backend configurado em settings.llm_backend)...")
    llm = get_llm()

    audit_logger = get_default_logger()

    print("4) Montando fluxo LangGraph...")
    workflow = build_workflow(retriever=retriever, llm=llm, audit_logger=audit_logger)

    print(f"5) Executando fluxo para paciente_id={paciente_id}, pergunta={pergunta!r}\n")
    resultado = workflow.invoke({"paciente_id": paciente_id, "pergunta_medico": pergunta})

    print("--- Exames pendentes ---")
    print(resultado.get("exames_pendentes") or "Nenhum")

    print("\n--- Alertas emitidos ---")
    print(resultado.get("alertas_emitidos") or "Nenhum")

    print("\n--- Fontes utilizadas (explainability) ---")
    print(resultado.get("fontes_rag") or "Nenhuma")

    print("\n--- Resposta final do assistente ---")
    print(resultado.get("resposta_final"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Demo do assistente médico virtual")
    parser.add_argument("--paciente-id", type=int, default=1)
    parser.add_argument(
        "--pergunta",
        type=str,
        default="Qual a conduta indicada para lactato elevado com suspeita de sepse?",
    )
    args = parser.parse_args()
    run_demo(args.paciente_id, args.pergunta)


if __name__ == "__main__":
    main()
