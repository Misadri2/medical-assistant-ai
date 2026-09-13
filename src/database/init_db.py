"""
Cria as tabelas e popula o banco SQLite com pacientes/exames sintéticos
para exercitar o fluxo do LangGraph (verificar exames pendentes, etc.).

Uso:
    python -m src.database.init_db
"""
from __future__ import annotations

from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from src.config import settings
from src.database.models import Alerta, Base, Exame, Paciente


def get_engine():
    if settings.database_url.startswith("sqlite:///"):
        db_path = Path(settings.database_url.replace("sqlite:///", "", 1))
        db_path.parent.mkdir(parents=True, exist_ok=True)
    return create_engine(settings.database_url, echo=False)


def init_db(seed: bool = True) -> None:
    engine = get_engine()
    Base.metadata.create_all(engine)

    if not seed:
        return

    with Session(engine) as session:
        # Evita duplicar seed se já existir dado.
        if session.query(Paciente).count() > 0:
            print("Banco já populado — pulando seed.")
            return

        p1 = Paciente(nome_iniciais="J.S.", idade=67, sexo="M")
        p2 = Paciente(nome_iniciais="M.A.", idade=45, sexo="F")
        p3 = Paciente(nome_iniciais="R.O.", idade=58, sexo="M")

        session.add_all([p1, p2, p3])
        session.flush()  # garante IDs

        session.add_all(
            [
                Exame(paciente_id=p1.id, tipo="lactato", status="concluido", valor_numerico=4.2),
                Exame(paciente_id=p1.id, tipo="hemocultura", status="pendente"),
                Exame(paciente_id=p2.id, tipo="troponina", status="concluido", valor_numerico=0.01),
                Exame(paciente_id=p2.id, tipo="ecg", status="concluido", resultado="sem supra de ST"),
                Exame(paciente_id=p3.id, tipo="potassio", status="concluido", valor_numerico=3.1),
                Exame(paciente_id=p3.id, tipo="gasometria", status="pendente"),
            ]
        )

        session.add(
            Alerta(
                paciente_id=p1.id,
                severidade="critico",
                mensagem="Lactato elevado (4.2 mmol/L) — suspeita de sepse, hemocultura pendente.",
            )
        )

        session.commit()
        print("Banco criado e populado com 3 pacientes sintéticos.")


if __name__ == "__main__":
    init_db()
