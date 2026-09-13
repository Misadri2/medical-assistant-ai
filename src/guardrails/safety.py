"""
Guardrails de segurança do assistente médico.

Requisito do desafio: "Definir limites de atuação do assistente para
evitar sugestões impróprias (ex.: nunca prescrever diretamente, sem
validação humana)".

Este módulo implementa duas camadas:
1. Validação de saída do LLM (output guardrail): garante que toda resposta
   com sugestão de conduta contenha o aviso de validação humana e não use
   linguagem de prescrição direta e imperativa.
2. Classificação de risco da pergunta (input guardrail): sinaliza perguntas
   fora do escopo do assistente (ex.: pedido explícito de dosagem para
   auto-administração, sem contexto de prontuário).
"""
from __future__ import annotations

import re
from dataclasses import dataclass

# Frases que indicam prescrição direta e imperativa, sem ressalva.
_PRESCRICAO_DIRETA = re.compile(
    r"\b(tome|administre|aplique|injete)\b\s+\d",
    re.IGNORECASE,
)

_AVISO_VALIDACAO = "requer validação de um médico responsável"

_FORA_DE_ESCOPO_KEYWORDS = [
    "sem receita",
    "para mim mesmo",
    "automedicação",
    "sem consultar médico",
]


@dataclass
class GuardrailResult:
    aprovado: bool
    motivo: str | None = None
    texto_ajustado: str | None = None


def check_input(question: str) -> GuardrailResult:
    """Bloqueia perguntas que buscam auto-medicação/uso fora do contexto clínico."""
    lowered = question.lower()
    for kw in _FORA_DE_ESCOPO_KEYWORDS:
        if kw in lowered:
            return GuardrailResult(
                aprovado=False,
                motivo=(
                    "Pergunta classificada como fora do escopo do assistente "
                    "(indício de uso para automedicação, sem supervisão clínica)."
                ),
            )
    return GuardrailResult(aprovado=True)


def check_output(response: str) -> GuardrailResult:
    """
    Garante que a resposta não prescreva diretamente sem ressalva, e que
    sempre contenha o aviso de necessidade de validação humana.
    """
    contains_direct_prescription = bool(_PRESCRICAO_DIRETA.search(response))
    contains_disclaimer = _AVISO_VALIDACAO in response.lower()

    if contains_direct_prescription and not contains_disclaimer:
        texto_ajustado = (
            response
            + f"\n\n[AVISO AUTOMÁTICO: esta sugestão {_AVISO_VALIDACAO} antes de qualquer execução.]"
        )
        return GuardrailResult(
            aprovado=True,
            motivo="Linguagem de prescrição direta detectada; aviso de validação humana adicionado.",
            texto_ajustado=texto_ajustado,
        )

    if not contains_disclaimer:
        texto_ajustado = (
            response
            + f"\n\n[AVISO AUTOMÁTICO: esta sugestão {_AVISO_VALIDACAO} antes de qualquer execução.]"
        )
        return GuardrailResult(
            aprovado=True,
            motivo="Aviso de validação humana ausente; adicionado automaticamente.",
            texto_ajustado=texto_ajustado,
        )

    return GuardrailResult(aprovado=True, texto_ajustado=response)
