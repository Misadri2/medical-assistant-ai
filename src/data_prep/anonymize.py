"""
Anonimização de dados clínicos antes do uso em fine-tuning ou RAG.

Dois backends:
- RegexAnonymizer: sem dependências externas, roda 100% offline. Usado por
  padrão em dev/teste e nesta atividade acadêmica (dados já são sintéticos).
- PresidioAnonymizer: usa Microsoft Presidio (NER via spaCy) para detecção
  mais robusta de PII em texto livre real. Requer `pip install
  presidio-analyzer presidio-anonymizer` e um modelo spaCy baixado
  (ex.: `python -m spacy download pt_core_news_lg`). É o backend recomendado
  para uso com dados reais do hospital.

A escolha do backend é feita por `settings.anonymizer_backend`.
"""
from __future__ import annotations

import re
from abc import ABC, abstractmethod

from src.config import settings

# Padrões simples de PII comuns em português (CPF, telefone, data de
# nascimento, nome precedido de "Sr./Sra./Dr.", e-mail).
_PATTERNS = {
    "CPF": re.compile(r"\b\d{3}\.\d{3}\.\d{3}-\d{2}\b"),
    "TELEFONE": re.compile(r"\b(?:\(\d{2}\)\s?)?\d{4,5}-?\d{4}\b"),
    "EMAIL": re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b"),
    "DATA": re.compile(r"\b\d{2}/\d{2}/\d{4}\b"),
    "NOME_COM_TITULO": re.compile(
        r"\b(?:Sr\.|Sra\.|Dr\.|Dra\.)\s+[A-ZÀ-Ý][a-zà-ÿ]+(?:\s+[A-ZÀ-Ý][a-zà-ÿ]+)*"
    ),
}


class BaseAnonymizer(ABC):
    @abstractmethod
    def anonymize(self, text: str) -> str:
        ...


class RegexAnonymizer(BaseAnonymizer):
    """Substitui padrões de PII por tags como [CPF], [TELEFONE], etc."""

    def anonymize(self, text: str) -> str:
        result = text
        for label, pattern in _PATTERNS.items():
            result = pattern.sub(f"[{label}]", result)
        return result


class PresidioAnonymizer(BaseAnonymizer):
    """
    Backend baseado em Microsoft Presidio. Import feito de forma lazy para
    não exigir a dependência quando não utilizado.
    """

    def __init__(self, language: str = "pt"):
        try:
            from presidio_analyzer import AnalyzerEngine
            from presidio_anonymizer import AnonymizerEngine
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "Presidio não instalado. Rode: pip install presidio-analyzer "
                "presidio-anonymizer e baixe um modelo spaCy compatível."
            ) from exc

        self._analyzer = AnalyzerEngine()
        self._anonymizer = AnonymizerEngine()
        self._language = language

    def anonymize(self, text: str) -> str:
        results = self._analyzer.analyze(text=text, language=self._language)
        anonymized = self._anonymizer.anonymize(text=text, analyzer_results=results)
        return anonymized.text


def get_anonymizer(backend: str | None = None) -> BaseAnonymizer:
    backend = backend or settings.anonymizer_backend
    if backend == "presidio":
        return PresidioAnonymizer()
    return RegexAnonymizer()
