"""Tests for the NLP server (symptom extraction, negation, scrubbing)."""
import os
import sys

sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..")))

from t3_agentic.nlp.server import analyze_text


def test_infectious_note():
    r = analyze_text("Patient de 64 ans, toux productive depuis 5 jours, "
                     "fievre a 38,9 C, douleur thoracique a l'inspiration")
    assert "toux" in r["entities"] and "fievre" in r["entities"]
    assert r["orientation"] == "infection respiratoire probable"
    assert r["suffisant"] is True


def test_negated_symptom_is_not_counted():
    r = analyze_text("Patient sans toux, avec fievre et dyspnee.")
    assert "toux" not in r["entities"]
    assert "dyspnee" in r["entities"]


def test_explicit_denial_is_non_infectious():
    r = analyze_text("Pas de toux, pas de fievre, pas de dyspnee.")
    assert r["orientation"] == "non infectieux"
    assert r["suffisant"] is True


def test_vague_note_is_insufficient():
    r = analyze_text("Le patient se sent un peu mieux aujourd'hui.")
    assert r["suffisant"] is False


def test_empty_note():
    r = analyze_text("")
    assert r["suffisant"] is False


def test_diagnostic_mentions_are_scrubbed():
    r = analyze_text("Diagnostic de pneumonie retenu. Toux et fievre.")
    assert r["redactions"] >= 1
    assert "pneumonie" not in r.get("orientation", "")


def test_english_note():
    r = analyze_text("Cough with fever and chest pain since three days.")
    assert "toux" in r["entities"]
    assert r["suffisant"] is True
