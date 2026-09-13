"""
Interface de inferência do LLM fine-tunado.

Dois backends:
- MockLLM: resposta determinística baseada no contexto recuperado, sem
  nenhuma dependência de modelo real. Usado em testes e nesta atividade
  quando não há um servidor Ollama disponível na máquina local.
- OllamaLLM: serve o modelo fine-tunado (LoRA já mesclado ou como
  Modelfile do Ollama) rodando localmente após o fine-tuning no Colab.
  Pressupõe que o adapter/modelo foi exportado do Colab (ver
  notebooks/fine_tuning_colab.ipynb) e importado no Ollama local com
  `ollama create medassist-llama3-8b -f Modelfile`.

A troca de backend é feita por `settings.llm_backend`.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

from src.config import settings

SYSTEM_PROMPT = (
    "Você é um assistente clínico de apoio à decisão. Responda SOMENTE com base "
    "no contexto de protocolos fornecido. NUNCA prescreva medicação diretamente "
    "sem indicar que requer validação de um médico responsável. Sempre cite a "
    "fonte (ID do protocolo) usada na resposta."
)


class BaseLLM(ABC):
    @abstractmethod
    def generate(self, question: str, context: str) -> str:
        ...


class MockLLM(BaseLLM):
    """
    Simula uma resposta do LLM fine-tunado combinando pergunta + contexto
    recuperado. Suficiente para exercitar o pipeline (RAG -> LangGraph ->
    guardrails -> auditoria) sem depender de GPU/Ollama.
    """

    def generate(self, question: str, context: str) -> str:
        if not context.strip():
            return (
                "Não encontrei protocolo interno relevante para essa pergunta. "
                "Recomendo consulta a um médico responsável antes de qualquer conduta."
            )
        # O contexto vem formatado como blocos "[Fonte: ID — titulo]\nconteudo",
        # separados por linha em branco. Usamos apenas o primeiro bloco (mais
        # relevante) para a resposta simulada.
        primeiro_bloco = context.split("\n\n")[0]
        linhas = primeiro_bloco.split("\n", 1)
        fonte = linhas[0].strip("[]")
        conteudo = linhas[1] if len(linhas) > 1 else ""
        return (
            f"Com base no protocolo interno ({fonte}), a conduta recomendada é: {conteudo[:400]}"
            "\n\n[Esta sugestão requer validação de um médico responsável antes de qualquer execução.]"
        )


class OllamaLLM(BaseLLM):
    def __init__(self, model_name: str | None = None, host: str | None = None):
        import ollama

        self._client = ollama.Client(host=host or settings.ollama_host)
        self._model_name = model_name or settings.ollama_model_name

    def generate(self, question: str, context: str) -> str:
        prompt = f"{SYSTEM_PROMPT}\n\nContexto:\n{context}\n\nPergunta: {question}\nResposta:"
        response = self._client.generate(model=self._model_name, prompt=prompt)
        return response["response"]


def get_llm() -> BaseLLM:
    if settings.llm_backend == "ollama":
        return OllamaLLM()
    return MockLLM()
