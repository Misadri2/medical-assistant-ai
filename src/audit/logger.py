"""
Logging de auditoria detalhado, exigido pelo desafio ("Implementar logging
detalhado para rastreamento e auditoria").

Cada interação do assistente gera um evento JSON estruturado (via
structlog) contendo: pergunta, fontes usadas na resposta (explainability),
resultado dos guardrails, e a resposta final. Os eventos são gravados em
um arquivo JSONL (um JSON por linha), formato fácil de consultar/exportar
para o relatório técnico.

A classe `AuditLogger` recebe o caminho do arquivo explicitamente (em vez
de configurar o structlog globalmente na importação do módulo), o que
facilita testes com arquivos temporários e evita handles de arquivo
"grudados" entre execuções.
"""
from __future__ import annotations

import datetime as dt
from pathlib import Path

import structlog

from src.config import settings


class AuditLogger:
    def __init__(self, log_path: str | None = None):
        self._log_path = Path(log_path or settings.audit_log_path)
        self._log_path.parent.mkdir(parents=True, exist_ok=True)
        self._file = open(self._log_path, "a", encoding="utf-8")

        self._logger = structlog.wrap_logger(
            structlog.PrintLogger(file=self._file),
            processors=[
                structlog.processors.TimeStamper(fmt="iso"),
                structlog.processors.JSONRenderer(ensure_ascii=False),
            ],
        )

    def log_interaction(
        self,
        *,
        pergunta: str,
        fontes: list[str],
        guardrail_input_aprovado: bool,
        guardrail_output_motivo: str | None,
        resposta_final: str,
        paciente_id: int | None = None,
    ) -> None:
        self._logger.info(
            "interacao_assistente",
            timestamp=dt.datetime.now(dt.UTC).isoformat(),
            pergunta=pergunta,
            paciente_id=paciente_id,
            fontes_utilizadas=fontes,
            guardrail_input_aprovado=guardrail_input_aprovado,
            guardrail_output_motivo=guardrail_output_motivo,
            resposta_final=resposta_final,
        )
        self._file.flush()

    def log_alerta_emitido(self, *, paciente_id: int, severidade: str, mensagem: str) -> None:
        self._logger.info(
            "alerta_emitido",
            timestamp=dt.datetime.now(dt.UTC).isoformat(),
            paciente_id=paciente_id,
            severidade=severidade,
            mensagem=mensagem,
        )
        self._file.flush()

    def close(self) -> None:
        self._file.close()


_default_logger: AuditLogger | None = None


def get_default_logger() -> AuditLogger:
    global _default_logger
    if _default_logger is None:
        _default_logger = AuditLogger()
    return _default_logger
