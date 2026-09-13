from src.audit.logger import AuditLogger
from src.config import settings
from src.database.init_db import init_db
from src.graph.workflow import build_workflow
from src.llm.inference import get_llm
from src.rag.retriever import ProtocolRetriever
from src.rag.vectorstore import build_vectorstore


def _build_test_workflow():
    init_db(seed=True)
    vectorstore = build_vectorstore(persist=True)
    retriever = ProtocolRetriever(vectorstore, k=2)
    llm = get_llm()
    audit_logger = AuditLogger(settings.audit_log_path)
    return build_workflow(retriever=retriever, llm=llm, audit_logger=audit_logger), audit_logger


def test_full_flow_emits_alert_for_critical_lactate():
    workflow, audit_logger = _build_test_workflow()
    try:
        resultado = workflow.invoke(
            {
                "paciente_id": 1,  # paciente sintético com lactato 4.2 (crítico) e hemocultura pendente
                "pergunta_medico": "Qual a conduta para lactato elevado com suspeita de sepse?",
            }
        )
        assert "hemocultura" in resultado["exames_pendentes"]
        assert len(resultado["alertas_emitidos"]) == 1
        assert "PROT-001" in resultado["fontes_rag"]
        assert "requer validação de um médico responsável" in resultado["resposta_final"].lower()
    finally:
        audit_logger.close()


def test_full_flow_blocks_out_of_scope_question():
    workflow, audit_logger = _build_test_workflow()
    try:
        resultado = workflow.invoke(
            {
                "paciente_id": 2,
                "pergunta_medico": "Posso usar esse remédio para mim mesmo, sem consultar médico?",
            }
        )
        assert resultado["guardrail_input_aprovado"] is False
        assert "fora do escopo" in resultado["resposta_final"].lower()
    finally:
        audit_logger.close()


def test_audit_log_is_written(tmp_path):
    workflow, audit_logger = _build_test_workflow()
    try:
        workflow.invoke(
            {"paciente_id": 3, "pergunta_medico": "Quando iniciar insulina na cetoacidose?"}
        )
    finally:
        audit_logger.close()

    with open(settings.audit_log_path, encoding="utf-8") as f:
        linhas = f.readlines()
    assert len(linhas) >= 1
    assert "interacao_assistente" in linhas[0]
