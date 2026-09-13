from src.guardrails import safety


def test_input_guardrail_blocks_self_medication():
    result = safety.check_input("Posso tomar esse remédio sem receita, para mim mesmo?")
    assert result.aprovado is False
    assert result.motivo is not None


def test_input_guardrail_allows_clinical_question():
    result = safety.check_input("Qual a conduta para lactato elevado com suspeita de sepse?")
    assert result.aprovado is True


def test_output_guardrail_adds_disclaimer_when_missing():
    resposta = "A conduta recomendada é iniciar antibiótico de amplo espectro."
    result = safety.check_output(resposta)
    assert result.aprovado is True
    assert "requer validação de um médico responsável" in result.texto_ajustado.lower()


def test_output_guardrail_flags_direct_prescription():
    resposta = "Tome 500mg de paracetamol agora."
    result = safety.check_output(resposta)
    assert "requer validação" in result.texto_ajustado.lower()
    assert result.motivo is not None


def test_output_guardrail_keeps_response_untouched_when_already_safe():
    resposta = (
        "Sugestão: considerar antibiótico de amplo espectro. "
        "Esta sugestão requer validação de um médico responsável antes de qualquer execução."
    )
    result = safety.check_output(resposta)
    assert result.texto_ajustado == resposta
