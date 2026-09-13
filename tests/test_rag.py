from src.rag.retriever import ProtocolRetriever
from src.rag.vectorstore import build_vectorstore, load_protocol_documents


def test_load_protocol_documents_returns_all_protocols():
    docs = load_protocol_documents()
    assert len(docs) == 6
    assert all("fonte_id" in d.metadata for d in docs)


def test_vectorstore_retrieves_relevant_protocol():
    vectorstore = build_vectorstore(persist=True)
    retriever = ProtocolRetriever(vectorstore, k=2)

    chunks = retriever.retrieve("conduta para lactato elevado suspeita de sepse")
    assert len(chunks) > 0
    fontes = [c.fonte_id for c in chunks]
    assert "PROT-001" in fontes  # protocolo de sepse


def test_format_context_with_sources_includes_fonte_id():
    vectorstore = build_vectorstore(persist=True)
    retriever = ProtocolRetriever(vectorstore, k=1)
    chunks = retriever.retrieve("crise hipertensiva")
    contexto = retriever.format_context_with_sources(chunks)
    assert "Fonte:" in contexto
