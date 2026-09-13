# Assistente Médico Virtual — Tech Challenge Fase 3 (Pós Tech IA)

Assistente virtual médico treinado com dados internos do hospital
(sintéticos, nesta atividade), capaz de responder dúvidas
clínicas com base em protocolos institucionais, verificar exames
pendentes de um paciente, emitir alertas para a equipe médica e manter
trilha de auditoria — tudo com guardrails de segurança (nunca prescreve
diretamente sem validação humana) e explainability (toda resposta cita a
fonte/protocolo utilizado).

## Sumário

- [Arquitetura](#arquitetura)
- [Estrutura do repositório](#estrutura-do-repositório)
- [Requisitos](#requisitos)
- [Instalação e execução local](#instalação-e-execução-local)
- [Fine-tuning no Google Colab](#fine-tuning-no-google-colab)
- [Rodando com o modelo fine-tunado (Ollama)](#rodando-com-o-modelo-fine-tunado-ollama)
- [Testes](#testes)
- [Guardrails e segurança](#guardrails-e-segurança)
- [Auditoria e explainability](#auditoria-e-explainability)
- [Como subir este repositório para o GitHub](#como-subir-este-repositório-para-o-github)
- [Decisões técnicas e limitações](#decisões-técnicas-e-limitações)

## Arquitetura

```
                         ┌─────────────────────┐
                         │   Google Colab       │
                         │  (fine-tuning GPU)   │
                         │  Unsloth + QLoRA      │
                         └──────────┬───────────┘
                                    │ exporta adapter LoRA / GGUF
                                    ▼
┌───────────────────────────────────────────────────────────────────┐
│                        Ambiente local (este repo)                  │
│                                                                     │
│   ┌────────────┐     ┌──────────────┐     ┌────────────────────┐  │
│   │  SQLite     │     │  ChromaDB     │     │  Ollama (opcional)  │  │
│   │ prontuários │     │  protocolos   │     │  LLM fine-tunado    │  │
│   │ (exames)    │     │  (RAG)        │     │  ou MockLLM (dev)   │  │
│   └─────┬──────┘     └──────┬───────┘     └──────────┬─────────┘  │
│         │                    │                          │           │
│         └──────────────┬─────┴──────────────┬───────────┘           │
│                         ▼                    ▼                       │
│                  ┌─────────────────────────────────┐                │
│                  │      Fluxo LangGraph              │                │
│                  │ check_pending_exams               │                │
│                  │  -> guardrail_input                │                │
│                  │  -> retrieve_context (RAG)          │                │
│                  │  -> generate_suggestion (LLM)        │                │
│                  │  -> guardrail_output                 │                │
│                  │  -> emit_alerts                        │                │
│                  │  -> audit_log                           │                │
│                  └─────────────────────────────────┘                │
│                                                                     │
└───────────────────────────────────────────────────────────────────┘
```

## Estrutura do repositório

```
medical-assistant-ai/
├── data/
│   ├── synthetic/          # protocolos, FAQs e modelos de documento sintéticos
│   └── processed/          # gerado em runtime: banco SQLite, Chroma, dataset de fine-tuning, log de auditoria
├── src/
│   ├── config.py           # configuração central (.env)
│   ├── data_prep/          # geração de dataset sintético + anonimização (regex/Presidio)
│   ├── database/           # modelos SQLAlchemy + seed do SQLite
│   ├── rag/                # embeddings, vector store (Chroma), retriever com fonte
│   ├── llm/                # inferência (MockLLM para dev, OllamaLLM para o modelo real)
│   ├── graph/               # estado, nós e workflow do LangGraph
│   ├── guardrails/          # regras de segurança (input/output)
│   ├── audit/                # logging estruturado (structlog) em JSONL
│   └── main.py               # CLI de demonstração ponta a ponta
├── notebooks/
│   └── fine_tuning_colab.ipynb   # fine-tuning Llama 3 8B com Unsloth + QLoRA
├── tests/                    # 18 testes automatizados (pytest)
├── docs/
│   └── relatorio_tecnico.md  # relatório técnico exigido na entrega
├── requirements.txt
├── .env.example
└── README.md
```

## Requisitos

- Python 3.10+
- (Opcional, para servir o modelo fine-tunado localmente) [Ollama](https://ollama.com/download) instalado
- (Opcional, para anonimização com Presidio) modelo spaCy: `python -m spacy download pt_core_news_lg`
- Conta no Google Colab (gratuita) para o fine-tuning
- Conta no Hugging Face (gratuita) para publicar o adapter LoRA treinado

## Instalação e execução local

```bash
# 1) Clonar o repositório
git clone https://github.com/SEU_USUARIO/SEU_REPOSITORIO.git
cd SEU_REPOSITORIO

# 2) Criar e ativar o ambiente virtual
python -m venv .venv
# Windows (PowerShell):
.venv\Scripts\Activate.ps1
# Linux/Mac:
source .venv/bin/activate

# 3) Instalar dependências
pip install -r requirements.txt

# 4) Copiar o arquivo de configuração
cp .env.example .env   # no Windows: copy .env.example .env

# 5) Gerar o dataset de fine-tuning (usado depois no Colab)
python -m src.data_prep.generate_synthetic_data

# 6) Rodar a demonstração completa (banco + RAG + LangGraph + guardrails + auditoria)
python -m src.main --paciente-id 1 --pergunta "Qual a conduta para lactato elevado com suspeita de sepse?"
```

## Fine-tuning no Google Colab

1. Abra `notebooks/fine_tuning_colab.ipynb` no Google Colab (upload direto
   ou "Open in Colab" a partir do GitHub, depois de subir o repositório).
2. Em **Runtime > Change runtime type**, selecione **GPU (T4)**.
3. Ajuste a variável `REPO_URL` na notebook para apontar para o seu
   repositório no GitHub (célula de clonagem).
4. Rode as células em ordem. O notebook:
   - instala Unsloth + dependências de fine-tuning;
   - clona o repositório e gera o dataset a partir dos JSONs sintéticos
     em `data/synthetic/`;
   - carrega o Llama 3 8B em 4-bit e aplica LoRA (QLoRA);
   - treina (SFT) com `TrainingArguments` ajustados para a T4 gratuita;
   - avalia qualitativamente (perguntas de treino e fora do treino) e
     quantitativamente (ROUGE-L contra respostas de referência);
   - publica o adapter LoRA no Hugging Face Hub;
   - exporta o modelo mesclado em GGUF e gera o `Modelfile` do Ollama.
5. Baixe a pasta `medassist_gguf/` e o `Modelfile` gerados ao final do
   notebook para a sua máquina local.

## Rodando com o modelo fine-tunado (Ollama)

```bash
# Na pasta onde estão o Modelfile e a pasta medassist_gguf/ baixados do Colab
ollama create medassist-llama3-8b -f Modelfile

# No .env do projeto, altere:
LLM_BACKEND=ollama
OLLAMA_MODEL_NAME=medassist-llama3-8b
OLLAMA_HOST=http://localhost:11434

# Rodar novamente a demonstração, agora com o modelo fine-tunado real:
python -m src.main --pergunta "Qual a conduta para lactato elevado com suspeita de sepse?"
```

## Testes

```bash
pytest -v
```

18 testes cobrindo: banco de dados/seed, anonimização (regex), RAG
(indexação e recuperação com fonte), guardrails de entrada/saída, e o
fluxo LangGraph completo (incluindo emissão de alerta para exame
crítico e bloqueio de pergunta fora de escopo).

## Guardrails e segurança

Implementados em `src/guardrails/safety.py`:

- **Guardrail de entrada:** bloqueia perguntas com indícios de uso para
  automedicação/fora do contexto clínico institucional (ex.: "sem
  receita", "para mim mesmo").
- **Guardrail de saída:** garante que toda sugestão de conduta clínica
  contenha, ao final, o aviso de que **requer validação de um médico
  responsável antes de qualquer execução** — inserido automaticamente
  caso a resposta do LLM não o contenha, e reforçado quando é detectada
  linguagem de prescrição direta e imperativa (ex.: "tome X mg").

## Auditoria e explainability

- Cada interação do assistente é registrada em
  `data/processed/audit_log.jsonl` (uma linha JSON por evento), via
  `structlog`, contendo: pergunta, paciente, fontes (protocolos)
  utilizadas na resposta, resultado dos guardrails e resposta final.
- Alertas emitidos para a equipe médica (ex.: exame crítico) também
  geram evento de auditoria próprio.
- **Explainability:** toda resposta do assistente é gerada a partir de
  chunks recuperados do RAG, e cada chunk carrega o metadado `fonte_id`
  do protocolo de origem — a resposta final referencia explicitamente
  qual protocolo embasou a sugestão (ex.: `PROT-001`).

## Como subir este repositório para o GitHub

```bash
cd "C:\Users\mismo\Área de Trabalho\Documentos\Pós IA\Repositórios Desafios Tech\medical-assistant-ai"

git init
git add .
git commit -m "Tech Challenge Fase 3: assistente médico com fine-tuning, RAG e LangGraph"

# Crie um repositório vazio no GitHub (via site) e depois:
git branch -M main
git remote add origin https://github.com/SEU_USUARIO/SEU_REPOSITORIO.git
git push -u origin main
```

> O `.gitignore` já exclui `.venv/`, `data/processed/` (banco, vector
> store e logs gerados em runtime) e o arquivo `.env` com segredos —
> apenas o código-fonte, os dados sintéticos de entrada e a
> documentação são versionados.

## Decisões técnicas e limitações

| Decisão | Motivo |
|---|---|
| Fine-tuning com **Unsloth + QLoRA** (4-bit) em vez de fine-tuning completo | Roda na GPU T4 gratuita do Colab, dentro dos limites de tempo/memória da atividade |
| Dados **100% sintéticos** (protocolos, FAQs, documentos) | O hospital real não existe nesta atividade acadêmica; a estrutura de dados é compatível com dados reais, bastando substituir os JSONs em `data/synthetic/` |
| `MockLLM` como backend padrão local | Permite rodar e testar 100% do pipeline (RAG, LangGraph, guardrails, auditoria) sem GPU nem servidor Ollama, e mantém os testes automatizados rápidos e determinísticos |
| `HashEmbeddings` como backend padrão de embedding | Evita dependência de download de modelo (sentence-transformers) para rodar os testes offline; trocável por `EMBEDDING_BACKEND=sentence-transformers` em produção |
| Anonimização com backend **regex** por padrão, **Presidio** opcional | Regex cobre os testes/dados sintéticos sem dependências pesadas; Presidio (NER) é o backend recomendado para dados clínicos reais em texto livre |
| Nunca prescrição direta sem validação humana | Requisito explícito do desafio — implementado como guardrail de saída, não apenas como instrução de prompt (defesa em profundidade) |

---
Projeto acadêmico desenvolvido para a Fase 3 da Pós Tech em IA (FIAP).
