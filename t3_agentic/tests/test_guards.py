"""Tests for the answer-level guards (provenance + non-diagnostic warning)."""
import os
import sys

sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..")))

from t3_agentic.agent.agent import _provenance_ok, _template_answer, _warning_ok
from t3_agentic.agent.prompts import WARNING

ALLOWED = '{"fusion": {"confidence": "faible"}, "image": {"p_pneumonia": 0.0605, ' \
          '"score": 0.9395, "inside_ratio": 0.177}, "heatmap": "gradcam_NORMAL_0.png"}'


def test_exact_numbers_pass():
    assert _provenance_ok("score 0.9395", ALLOWED)


def test_rounded_numbers_pass():
    assert _provenance_ok("score 0.94", ALLOWED)
    assert _provenance_ok("score 0,94", ALLOWED)


def test_percentage_form_passes():
    assert _provenance_ok("confiance de 94 %", ALLOWED)


def test_filename_digits_are_not_numbers():
    # "gradcam_NORMAL_0.png" must not be parsed as the number 0
    assert _provenance_ok("voir gradcam_NORMAL_0.png", ALLOWED)


def test_invented_number_fails():
    assert not _provenance_ok("probabilite de 0.72", ALLOWED)
    assert not _provenance_ok("3 lesions", ALLOWED)


def test_warning_detection_is_typo_tolerant():
    assert _warning_ok("Prototype pédagogique, ne constitue pas un diagnostic médical.")
    assert _warning_ok("Prototype pedagogique, ne constitue pas un diagnostic medical")
    assert not _warning_ok("Le patient va bien.")


def test_canonical_warning_used():
    assert "ne constitue pas un diagnostic médical" in WARNING


def test_uncertain_image_answer_reports_prediction_and_abstention():
    vision = {
        "label": "normal", "score": 0.5647,
        "p_pneumonia_calibrated": 0.4353, "label_threshold": 0.685,
        "inside_ratio": 0.82, "suspect": False,
    }
    fusion = {"refused": True, "reason": "image seule, image incertaine",
              "confidence": "aucune", "coherence": "insuffisant"}
    answer = _template_answer(fusion, vision, None)
    assert "normal" in answer
    assert "43.5%" in answer and "56.5%" in answer and "68.5%" in answer
    assert "s'abstient" in answer
    assert "Aucun symptome n'a ete indique dans une note" in answer
    assert "ne signifie pas que le patient est sans symptomes" in answer
    assert "ne localise ni ne confirme une lesion" in answer
    assert "ne constitue pas un diagnostic médical" in answer


def test_note_symptoms_are_not_attributed_to_image():
    fusion = {"refused": False, "final": "pneumonie_suspectee",
              "confidence": "faible", "coherence": "texte_seule"}
    note = {"entities": ["toux", "fievre"], "denied": [],
            "orientation": "infection respiratoire probable", "suffisant": True}
    answer = _template_answer(fusion, None, note)
    assert "Symptomes extraits de la note (pas de l'image) : toux, fievre" in answer


def test_pneumonia_prediction_does_not_invent_symptoms():
    vision = {"label": "pneumonie", "score": 0.95,
              "p_pneumonia_calibrated": 0.95, "label_threshold": 0.685}
    fusion = {"refused": False, "final": "pneumonie_suspectee",
              "confidence": "haute", "coherence": "image_seule"}
    answer = _template_answer(fusion, vision, None)
    assert "Symptomes cliniques non renseignes" in answer
    assert "Ajoutez une note clinique" in answer
    assert "toux" not in answer and "fievre" not in answer
