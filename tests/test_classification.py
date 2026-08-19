"""Tests for Agent 2: classification JSON parsing and fallbacks."""

import json

import pytest

from app.agents.classification import (
    ClassificationAgent,
    ClassificationResult,
    heuristic_classify,
    parse_classification_json,
)
from app.llm.client import NullClient


class MockLLM:
    """Scripted LLM double returning a canned response."""

    def __init__(self, response: str):
        self.response = response
        self.calls = []

    def complete(self, system: str, user: str, json_mode: bool = False) -> str:
        self.calls.append({"system": system, "user": user, "json_mode": json_mode})
        return self.response


SAMPLE_RESPONSE = json.dumps(
    {
        "document_class": "act_vanzare_cumparare",
        "confidence": 0.92,
        "fields": {
            "parti": ["POPESCU ION", "IONESCU MARIA"],
            "numar_cadastral": "123456/1/2",
            "data": "12.03.2024",
            "adresa": "Str. Victoriei nr. 10, Bucuresti",
            "notar": "BPN Georgescu Ana",
            "valoare": "150000 EUR",
        },
        "tags": ["<parte>", "<numar_cadastral>", "<data>", "<valoare>"],
        "tagged_text": "Intre <parte>POPESCU ION</parte> si <parte>IONESCU MARIA</parte>",
    }
)


def test_parse_valid_json():
    result = parse_classification_json(SAMPLE_RESPONSE)
    assert result.document_class == "act_vanzare_cumparare"
    assert result.class_label == "Act de vanzare-cumparare"
    assert result.confidence == pytest.approx(0.92)
    assert result.fields["numar_cadastral"] == "123456/1/2"
    assert "<parte>" in result.tags
    assert result.agent_used == "llm"


def test_parse_json_inside_code_fence():
    raw = f"Iata rezultatul:\n```json\n{SAMPLE_RESPONSE}\n```\nSper ca ajuta."
    result = parse_classification_json(raw)
    assert result.document_class == "act_vanzare_cumparare"


def test_parse_unknown_class_falls_back_to_alt_document():
    raw = json.dumps({"document_class": "factura", "confidence": 0.5})
    result = parse_classification_json(raw)
    assert result.document_class == "alt_document"
    assert result.class_label == "Alt document"


def test_parse_invalid_json_raises():
    with pytest.raises(ValueError):
        parse_classification_json("nu exista JSON aici")


def test_confidence_is_clamped():
    raw = json.dumps({"document_class": "procura", "confidence": 7.5})
    assert parse_classification_json(raw).confidence == 1.0


def test_agent_uses_llm_and_parses():
    agent = ClassificationAgent(MockLLM(SAMPLE_RESPONSE))
    result = agent.classify("Contract de vanzare-cumparare intre parti ...")
    assert result.document_class == "act_vanzare_cumparare"
    assert result.agent_used == "llm"


def test_agent_falls_back_to_heuristic_on_llm_error():
    class BrokenLLM:
        def complete(self, system, user, json_mode=False):
            raise RuntimeError("llm down")

    agent = ClassificationAgent(BrokenLLM())
    result = agent.classify("CERTIFICAT DE URBANISM eliberat de primarie")
    assert result.document_class == "certificat_urbanism"
    assert result.agent_used == "heuristic"


def test_null_client_uses_heuristic():
    agent = ClassificationAgent(NullClient())
    result = agent.classify("EXTRAS DE CARTE FUNCIARA nr. 405123 ANCPI")
    assert result.document_class == "extras_carte_funciara"


def test_heuristic_plan_detection():
    result = heuristic_classify("Plan cadastral parcela A1 tarla 22", has_plan_region=True)
    assert result.document_class == "plan_cadastral"


def test_heuristic_unknown_document():
    result = heuristic_classify("text fara cuvinte relevante xyz")
    assert result.document_class == "alt_document"
    assert result.confidence < 0.5
