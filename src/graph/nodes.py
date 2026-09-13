"""
Nós do fluxo LangGraph do assistente médico.

Cada função recebe o `AssistantState` atual e retorna um dicionário parcial
com as chaves que deseja atualizar (padrão do LangGraph).
"""
from __future__ import annotations

from sqlalchemy.orm import Session

from src.audit.logger import AuditLogger
from src.database.init_db import get_engine
from src.database.models import Alerta, Exame
from src.graph.state import AssistantState
from src.guardrails import safety
from src.llm.inference import BaseLLM
from src.rag.retriever import ProtocolRetriever


def make_check_pending_exams_node():
    def check_pending_exams(state: AssistantState) -> dict:
        engine = get_engine()
        with Session(engine) as session:
            exames = (
                session.query(Exame)
                .filter(Exame.paciente_id == state["paciente_id"])
                .all()
            )
            pendentes = [e.tipo for e in exames if e.status == "pendente"]
            criticos = [
                {"tipo": e.tipo, "valor": e.valor_numerico}
                for e in exames
                if e.tipo == "lactato" and e.valor_numerico and e.valor_numerico >= 4.0
            ]
        return {"exames_pendentes": pendentes, "exames_criticos": criticos}

    return check_pending_exams


def make_guardrail_input_node():
    def guardrail_input(state: AssistantState) -> dict:
        result = safety.check_input(state["pergunta_medico"])
        return {
            "guardrail_input_aprovado": result.aprovado,
            "guardrail_input_motivo": result.motivo,
        }

    return guardrail_input


def make_retrieve_context_node(retriever: ProtocolRetriever):
    def retrieve_context(state: AssistantState) -> dict:
        chunks = retriever.retrieve(state["pergunta_medico"])
        contexto = retriever.format_context_with_sources(chunks)
        fontes = [c.fonte_id for c in chunks]
        return {"contexto_rag": contexto, "fontes_rag": fontes}

    return retrieve_context


def make_generate_suggestion_node(llm: BaseLLM):
    def generate_suggestion(state: AssistantState) -> dict:
        if not state.get("guardrail_input_aprovado", True):
            return {
                "sugestao_llm": (
                    "Pergunta fora do escopo de apoio clínico deste assistente. "
                    "Consulte diretamente um médico responsável."
                )
            }
        sugestao = llm.generate(state["pergunta_medico"], state.get("contexto_rag", ""))
        return {"sugestao_llm": sugestao}

    return generate_suggestion


def guardrail_output_node(state: AssistantState) -> dict:
    result = safety.check_output(state["sugestao_llm"])
    return {
        "resposta_final": result.texto_ajustado or state["sugestao_llm"],
        "guardrail_output_motivo": result.motivo,
    }


def make_emit_alerts_node():
    def emit_alerts(state: AssistantState) -> dict:
        alertas_emitidos = []
        if state.get("exames_criticos"):
            engine = get_engine()
            with Session(engine) as session:
                for exame_critico in state["exames_criticos"]:
                    msg = (
                        f"Valor crítico de {exame_critico['tipo']} "
                        f"({exame_critico['valor']}) detectado — equipe médica notificada."
                    )
                    session.add(
                        Alerta(
                            paciente_id=state["paciente_id"],
                            severidade="critico",
                            mensagem=msg,
                        )
                    )
                    alertas_emitidos.append(msg)
                session.commit()
        return {"alertas_emitidos": alertas_emitidos}

    return emit_alerts


def make_audit_log_node(audit_logger: AuditLogger):
    def audit_log(state: AssistantState) -> dict:
        audit_logger.log_interaction(
            pergunta=state["pergunta_medico"],
            fontes=state.get("fontes_rag", []),
            guardrail_input_aprovado=state.get("guardrail_input_aprovado", True),
            guardrail_output_motivo=state.get("guardrail_output_motivo"),
            resposta_final=state.get("resposta_final", ""),
            paciente_id=state.get("paciente_id"),
        )
        for alerta in state.get("alertas_emitidos", []):
            audit_logger.log_alerta_emitido(
                paciente_id=state["paciente_id"], severidade="critico", mensagem=alerta
            )
        return {}

    return audit_log
