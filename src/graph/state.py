"""
Estado compartilhado entre os nós do fluxo LangGraph.

O fluxo implementa o cenário descrito no desafio: "ao receber informações
sobre um paciente, o sistema possa acionar diferentes etapas, como
verificar exames pendentes, sugerir tratamentos e emitir alertas para a
equipe médica".
"""
from __future__ import annotations

from typing import TypedDict


class AssistantState(TypedDict, total=False):
    paciente_id: int
    pergunta_medico: str

    exames_pendentes: list[str]
    exames_criticos: list[dict]

    contexto_rag: str
    fontes_rag: list[str]

    sugestao_llm: str
    guardrail_input_aprovado: bool
    guardrail_input_motivo: str | None
    guardrail_output_motivo: str | None

    alertas_emitidos: list[str]
    resposta_final: str
