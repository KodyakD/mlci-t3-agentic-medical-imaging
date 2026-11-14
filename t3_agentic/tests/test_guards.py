"""Tests for the answer-level guards (provenance + non-diagnostic warning)."""
import os
import sys

sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..")))

from t3_agentic.agent.agent import _provenance_ok, _warning_ok
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
