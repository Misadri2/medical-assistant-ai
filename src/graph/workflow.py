"""
Monta o grafo LangGraph do assistente médico.

Fluxo:
  START
    -> check_pending_exams   (verifica exames pendentes/críticos do paciente)
    -> guardrail_input       (valida se a pergunta está dentro do escopo)
    -> retrieve_context      (RAG: busca protocolos relevantes com fonte)
    -> generate_suggestion   (LLM gera sugestão com base no contexto)
    -> guardrail_output      (garante aviso de validação humana)
    -> emit_alerts           (emite alertas para exames críticos, se houver)
    -> audit_log             (grava evento de auditoria)
  END
"""
from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from src.audit.logger import AuditLogger
from src.graph import nodes
from src.graph.state import AssistantState
from src.llm.inference import BaseLLM
from src.rag.retriever import ProtocolRetriever


def build_workflow(
    retriever: ProtocolRetriever,
    llm: BaseLLM,
    audit_logger: AuditLogger,
):
    graph = StateGraph(AssistantState)

    graph.add_node("check_pending_exams", nodes.make_check_pending_exams_node())
    graph.add_node("guardrail_input", nodes.make_guardrail_input_node())
    graph.add_node("retrieve_context", nodes.make_retrieve_context_node(retriever))
    graph.add_node("generate_suggestion", nodes.make_generate_suggestion_node(llm))
    graph.add_node("guardrail_output", nodes.guardrail_output_node)
    graph.add_node("emit_alerts", nodes.make_emit_alerts_node())
    graph.add_node("audit_log", nodes.make_audit_log_node(audit_logger))

    graph.add_edge(START, "check_pending_exams")
    graph.add_edge("check_pending_exams", "guardrail_input")
    graph.add_edge("guardrail_input", "retrieve_context")
    graph.add_edge("retrieve_context", "generate_suggestion")
    graph.add_edge("generate_suggestion", "guardrail_output")
    graph.add_edge("guardrail_output", "emit_alerts")
    graph.add_edge("emit_alerts", "audit_log")
    graph.add_edge("audit_log", END)

    return graph.compile()
