# Relatório Técnico — Assistente Médico Virtual (Tech Challenge Fase 3)

## 1. Visão geral

Este projeto implementa um assistente virtual médico treinado com dados
internos do hospital (sintéticos, para fins acadêmicos), capaz de:

- responder dúvidas clínicas de médicos com base em protocolos internos
  (via fine-tuning + RAG);
- consultar registros estruturados de pacientes (exames pendentes,
  valores críticos);
- acionar um fluxo automatizado de decisão (LangGraph) que verifica
  exames, sugere condutas e emite alertas para a equipe médica;
- operar dentro de limites de segurança definidos (guardrails) e manter
  trilha de auditoria com explainability.
  https://huggingface.co/Misadri1984/medassist-lora-adapter

## 2. Processo de fine-tuning

**Modelo base:** Llama 3 8B (`unsloth/llama-3-8b-bnb-4bit`), escolhido
por ser open-weight, ter bom desempenho em tarefas de QA e caber em
4-bit na GPU T4 gratuita do Google Colab.

**Técnica:** QLoRA (Low-Rank Adaptation sobre pesos quantizados em
4-bit), via biblioteca Unsloth, que otimiza o uso de memória e velocidade
de treino em relação ao fine-tuning completo ou a um LoRA "puro" sem
quantização.

**Hiperparâmetros utilizados** (ver `notebooks/fine_tuning_colab.ipynb`):

| Parâmetro | Valor | Justificativa |
|---|---|---|
| `r` (rank do LoRA) | 16 | Equilíbrio entre capacidade de adaptação e uso de memória para um dataset pequeno |
| `lora_alpha` | 16 | Igual a `r`, escala neutra recomendada pela documentação do Unsloth |
| `target_modules` | q/k/v/o_proj, gate/up/down_proj | Cobre projeções de atenção e MLP, maximizando a capacidade de adaptação |
| `learning_rate` | 2e-4 | Padrão recomendado para QLoRA em datasets pequenos |
| `num_train_epochs` | 3 | Dataset sintético é pequeno (< 20 exemplos nesta atividade); mais épocas ajudam a fixar o padrão de resposta institucional |
| `per_device_train_batch_size` | 2 (com `gradient_accumulation_steps=4`, batch efetivo 8) | Cabe na memória da T4 com sequência de até 2048 tokens |

**Dados de treino:** gerados por
`src/data_prep/generate_synthetic_data.py`, combinando:
- FAQs de médicos (`data/synthetic/faq_medicos.json`);
- Protocolos institucionais resumidos como instrução de resumo
  (`data/synthetic/protocolos_hospitalares.json`);
- Modelos de documentos internos — laudos, receitas, procedimentos
  (`data/synthetic/modelos_documentos.json`).

Cada exemplo é formatado no template Alpaca (instrução / entrada /
resposta), padrão amplamente compatível com fine-tuning de LLMs
instrucionados como o Llama 3.

**Anonimização:** implementada em `src/data_prep/anonymize.py`, com dois
backends — regex (offline, usado nos dados sintéticos desta atividade,
que já não contêm PII real) e Microsoft Presidio (NER, recomendado para
dados clínicos reais em texto livre, com detecção de nomes, CPF,
telefone, e-mail e datas).

**Avaliação do modelo:**
- *Qualitativa:* comparação da resposta gerada para (a) uma pergunta
  presente no conjunto de treino (avalia memorização do padrão
  institucional) e (b) uma pergunta fora do conjunto de treino (avalia
  generalização a um protocolo correlato).
- *Quantitativa:* ROUGE-L entre a resposta gerada e a resposta de
  referência do protocolo, para um pequeno conjunto de validação
  hold-out (ver célula final do notebook). *(Preencher aqui os valores
  obtidos após rodar o notebook: ROUGE-L médio = \_\_\_\_.)*

> **Nota:** por se tratar de um dataset sintético e pequeno (gerado para
> fins de demonstração acadêmica do pipeline), os resultados
> quantitativos têm valor ilustrativo do processo, não de validação
> clínica do modelo.

## 3. Descrição do assistente médico

O assistente combina três componentes:

1. **RAG (Retrieval-Augmented Generation):** os protocolos hospitalares
   são indexados em uma base vetorial (ChromaDB). Ao receber uma
   pergunta, o sistema recupera os chunks mais similares e os injeta no
   prompt do LLM, junto com a identificação da fonte (`fonte_id`) —
   garantindo que a resposta seja rastreável a um protocolo específico
   (`src/rag/`).
2. **LLM fine-tunado:** gera a resposta em linguagem natural a partir do
   contexto recuperado, servido localmente via Ollama após o fine-tuning
   no Colab (`src/llm/inference.py`). Em ambiente de desenvolvimento/
   teste, um `MockLLM` determinístico substitui o modelo real, permitindo
   validar o pipeline sem GPU.
3. **Fluxo de decisão (LangGraph):** orquestra as etapas do atendimento
   a uma pergunta clínica sobre um paciente específico — descrito em
   detalhe na seção 4.

## 4. Diagrama do fluxo LangChain/LangGraph

```
START
  │
  ▼
check_pending_exams ──► consulta SQLite: exames pendentes e valores críticos do paciente
  │
  ▼
guardrail_input ──► bloqueia perguntas fora de escopo (ex.: automedicação)
  │
  ▼
retrieve_context ──► RAG: busca protocolos relevantes na base Chroma (com fonte)
  │
  ▼
generate_suggestion ──► LLM fine-tunado gera sugestão com base no contexto recuperado
  │
  ▼
guardrail_output ──► garante aviso de validação humana / remove prescrição direta sem ressalva
  │
  ▼
emit_alerts ──► se exame crítico (ex.: lactato >= 4.0), registra alerta para a equipe médica
  │
  ▼
audit_log ──► grava evento estruturado (pergunta, fontes, guardrails, resposta) em JSONL
  │
  ▼
END
```

Implementação: `src/graph/state.py` (estado compartilhado),
`src/graph/nodes.py` (cada etapa como função pura sobre o estado) e
`src/graph/workflow.py` (composição do grafo com `langgraph.graph.StateGraph`).

## 5. Segurança e validação

- **Limites de atuação:** o assistente nunca prescreve medicação/conduta
  de forma imperativa e definitiva — toda resposta é tratada como
  *sugestão* e recebe, automaticamente, o aviso de que requer validação
  de um médico responsável (`src/guardrails/safety.py`). Perguntas que
  sugerem uso do assistente para automedicação são bloqueadas antes de
  chegar ao LLM.
- **Logging e auditoria:** todo evento de interação e todo alerta emitido
  é gravado em `data/processed/audit_log.jsonl`, com timestamp, pergunta,
  paciente, fontes utilizadas e resultado dos guardrails — permitindo
  auditoria posterior de qualquer decisão sugerida pelo sistema.
- **Explainability:** a resposta final sempre referencia o(s) protocolo(s)
  (`fonte_id`) usados pelo RAG para compor a sugestão, evitando respostas
  "caixa-preta".

## 6. Avaliação dos resultados

- O pipeline completo (banco → RAG → LLM → guardrails → alertas →
  auditoria) foi validado ponta a ponta com 18 testes automatizados
  (`pytest`), cobrindo: seed e idempotência do banco, recuperação correta
  de protocolo relevante via RAG, bloqueio de pergunta fora de escopo,
  inserção automática do aviso de validação humana e emissão de alerta
  para exame crítico (lactato elevado), com geração do evento de
  auditoria correspondente.
- Limitação principal: os dados usados (protocolos, FAQs, exames) são
  sintéticos e em pequena quantidade — suficientes para demonstrar a
  arquitetura e o funcionamento correto do pipeline, mas não
  representativos da variedade e do volume de um hospital real. Uma
  evolução natural do projeto seria treinar com o dataset público
  PubMedQA/MedQuAD (sugeridos no enunciado) combinados aos protocolos
  reais do hospital, devidamente anonimizados via Presidio.

## 7. Como reproduzir

Ver README.md do repositório, seções "Instalação e execução local" e
"Fine-tuning no Google Colab".
