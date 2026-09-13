"""
Modelos de banco de dados relacional (SQLite) que simulam os registros
estruturados de prontuário eletrônico consultados pelo LangChain/LangGraph.

Em produção, isto seria substituído por uma conexão ao sistema real de
prontuário eletrônico do hospital (HL7/FHIR, banco proprietário, etc.).
Para esta atividade acadêmica, usamos SQLite com dados sintéticos.
"""
from __future__ import annotations

import datetime as dt

from sqlalchemy import DateTime, Float, ForeignKey, String
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class Paciente(Base):
    __tablename__ = "pacientes"

    id: Mapped[int] = mapped_column(primary_key=True)
    nome_iniciais: Mapped[str] = mapped_column(String(10))  # dado já anonimizado
    idade: Mapped[int] = mapped_column()
    sexo: Mapped[str] = mapped_column(String(1))

    exames: Mapped[list["Exame"]] = relationship(back_populates="paciente")
    alertas: Mapped[list["Alerta"]] = relationship(back_populates="paciente")


class Exame(Base):
    __tablename__ = "exames"

    id: Mapped[int] = mapped_column(primary_key=True)
    paciente_id: Mapped[int] = mapped_column(ForeignKey("pacientes.id"))
    tipo: Mapped[str] = mapped_column(String(50))
    status: Mapped[str] = mapped_column(String(20))  # "pendente" | "concluido"
    resultado: Mapped[str | None] = mapped_column(String(500), nullable=True)
    valor_numerico: Mapped[float | None] = mapped_column(Float, nullable=True)
    solicitado_em: Mapped[dt.datetime] = mapped_column(DateTime, default=lambda: dt.datetime.now(dt.UTC))

    paciente: Mapped["Paciente"] = relationship(back_populates="exames")


class Alerta(Base):
    __tablename__ = "alertas"

    id: Mapped[int] = mapped_column(primary_key=True)
    paciente_id: Mapped[int] = mapped_column(ForeignKey("pacientes.id"))
    severidade: Mapped[str] = mapped_column(String(20))  # "info" | "atencao" | "critico"
    mensagem: Mapped[str] = mapped_column(String(500))
    criado_em: Mapped[dt.datetime] = mapped_column(DateTime, default=lambda: dt.datetime.now(dt.UTC))

    paciente: Mapped["Paciente"] = relationship(back_populates="alertas")
