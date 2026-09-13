"""
Gera exemplos sintéticos adicionais de perguntas/respostas clínicas para
compor o dataset de fine-tuning, combinando os protocolos e FAQs
já existentes em `data/synthetic/`.

O objetivo não é ter um dataset médico realista/validado clinicamente,
mas sim ter uma massa de dados NO FORMATO CORRETO para exercitar o
pipeline completo (fine-tuning, RAG, avaliação) sem depender de dados
reais do hospital, que não estão disponíveis nesta atividade acadêmica.

Uso:
    python -m src.data_prep.generate_synthetic_data
"""
from __future__ import annotations

import json
from pathlib import Path

SYNTHETIC_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "synthetic"
PROCESSED_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "processed"


def _load(name: str) -> list[dict]:
    with open(SYNTHETIC_DIR / name, encoding="utf-8") as f:
        return json.load(f)


def build_finetuning_dataset() -> list[dict]:
    """
    Converte protocolos + FAQs para o formato instrução/resposta usado no
    fine-tuning (compatível com o formato esperado pelo TRL/SFTTrainer:
    {"instruction": ..., "input": ..., "output": ...}).
    """
    protocolos = _load("protocolos_hospitalares.json")
    faqs = _load("faq_medicos.json")
    documentos = _load("modelos_documentos.json")

    exemplos = []

    for faq in faqs:
        exemplos.append(
            {
                "instruction": faq["pergunta"],
                "input": "",
                "output": faq["resposta"],
                "fonte": faq["fonte_protocolo"],
            }
        )

    for prot in protocolos:
        exemplos.append(
            {
                "instruction": f"Resuma o {prot['titulo']} em uma orientação prática para a equipe médica.",
                "input": "",
                "output": prot["conteudo"],
                "fonte": prot["id"],
            }
        )

    for doc in documentos:
        exemplos.append(
            {
                "instruction": f"Gere um {doc['tipo']} seguindo o modelo interno: {doc['titulo']}.",
                "input": "",
                "output": doc["conteudo"],
                "fonte": doc["id"],
            }
        )

    return exemplos


def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    dataset = build_finetuning_dataset()
    out_path = PROCESSED_DIR / "finetuning_dataset.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for row in dataset:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Dataset de fine-tuning gerado: {out_path} ({len(dataset)} exemplos)")


if __name__ == "__main__":
    main()
