"""Tests for the deterministic modality gate (no LLM involved)."""
import os
import sys

sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..")))

from t3_agentic.agent.gate import (allowed_tools, complete_tools, fallback_route,
                                   required_tools, validate_tool_choice)


def test_text_only_never_exposes_cnn():
    assert "classify_image" not in allowed_tools(False, True)
    assert "explain_image" not in allowed_tools(False, True)
    assert "analyze_text" in allowed_tools(False, True)


def test_image_only_exposes_vision_tools():
    assert "classify_image" in allowed_tools(True, False)
    assert "explain_image" in allowed_tools(True, False)
    assert "analyze_text" not in allowed_tools(True, False)


def test_both_exposes_everything():
    assert "classify_image" in allowed_tools(True, True)
    assert "explain_image" in allowed_tools(True, True)
    assert "analyze_text" in allowed_tools(True, True)


def test_no_modality_falls_back_to_the_resource():
    # no image, no note -> the agent may read the MCP resource (knowledge question)
    assert "get_guideline" in allowed_tools(False, False)
    assert "search_guidelines" in allowed_tools(False, False)
    assert required_tools(False, False) == set()


def test_empty_note_is_still_the_text_modality():
    # an empty note is NOT "no note": the NLP tool must answer "note vide"
    assert "analyze_text" in allowed_tools(False, True)


def test_validate_drops_forbidden_and_dedups():
    proposed = ["analyze_text", "classify_image", "analyze_text"]
    assert validate_tool_choice(proposed, {"analyze_text"}) == ["analyze_text"]


def test_harness_completion_adds_missing_required():
    assert complete_tools(["analyze_text"], {"analyze_text", "classify_image"}) == ["classify_image"]


def test_fallback_route_uses_modalities_first():
    assert fallback_route("que dit-on ?", True, True) == [
        "classify_image", "explain_image", "analyze_text"]
    assert fallback_route("analyse", True, False) == ["classify_image", "explain_image"]
    assert fallback_route("analyse", False, True) == ["analyze_text"]
