"""Tests for the resources server content."""
import os
import sys

sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..")))

from t3_agentic.resources.content import (EXAMPLES, GUIDELINES_PNEUMONIA,
                                          PROTOTYPE_WARNING)


def test_warning_is_explicit():
    assert "Prototype" in PROTOTYPE_WARNING
    assert "diagnostic" in PROTOTYPE_WARNING.lower()


def test_guidelines_mention_contradiction():
    assert "contredisent" in GUIDELINES_PNEUMONIA.lower()


def test_examples_cover_three_routes():
    assert {e["route"] for e in EXAMPLES} == {"image", "text", "both"}
