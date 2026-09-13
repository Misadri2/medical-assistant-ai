from src.data_prep.anonymize import RegexAnonymizer, get_anonymizer


def test_regex_anonymizer_masks_cpf():
    anon = RegexAnonymizer()
    text = "Paciente com CPF 123.456.789-00 foi internado."
    result = anon.anonymize(text)
    assert "123.456.789-00" not in result
    assert "[CPF]" in result


def test_regex_anonymizer_masks_email_and_phone():
    anon = RegexAnonymizer()
    text = "Contato: joao@example.com ou (11) 91234-5678."
    result = anon.anonymize(text)
    assert "joao@example.com" not in result
    assert "[EMAIL]" in result
    assert "[TELEFONE]" in result


def test_regex_anonymizer_masks_named_title():
    anon = RegexAnonymizer()
    text = "O paciente foi atendido pelo Dr. Carlos Andrade na emergência."
    result = anon.anonymize(text)
    assert "Carlos Andrade" not in result
    assert "[NOME_COM_TITULO]" in result


def test_get_anonymizer_default_backend():
    anon = get_anonymizer()
    assert isinstance(anon, RegexAnonymizer)
