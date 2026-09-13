from sqlalchemy.orm import Session

from src.database.init_db import get_engine, init_db
from src.database.models import Exame, Paciente


def test_init_db_creates_and_seeds():
    init_db(seed=True)
    engine = get_engine()
    with Session(engine) as session:
        pacientes = session.query(Paciente).all()
        assert len(pacientes) == 3

        exames = session.query(Exame).all()
        assert len(exames) > 0


def test_init_db_is_idempotent():
    init_db(seed=True)
    init_db(seed=True)  # não deve duplicar
    engine = get_engine()
    with Session(engine) as session:
        assert session.query(Paciente).count() == 3


def test_pending_exams_query():
    init_db(seed=True)
    engine = get_engine()
    with Session(engine) as session:
        pendentes = (
            session.query(Exame)
            .filter(Exame.paciente_id == 1, Exame.status == "pendente")
            .all()
        )
        assert any(e.tipo == "hemocultura" for e in pendentes)
