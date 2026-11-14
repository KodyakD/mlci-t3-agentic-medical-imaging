"""Tests for the resources server (content, prompts, retrieval)."""
import os
import sys

sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..")))

from t3_agentic.resources.content import (EXAMPLES, GUIDELINES_PNEUMONIA,
                                          PROTOTYPE_WARNING)


def test_the_mcp_prompt_is_fetchable_and_uses_only_valid_roles():
    # MCP spec: PromptMessage roles must be 'user' or 'assistant' (never 'system')
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "resources_server",
        os.path.join(os.path.dirname(__file__), "..", "resources", "server.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    msgs = mod.cautious_radiology_assistant("Que peut-on dire ?")
    assert msgs, "prompt returned nothing"
    roles = {m["role"] for m in msgs}
    assert roles <= {"user", "assistant"}, f"invalid roles {roles}"
    assert any("Prototype" in m["content"] for m in msgs)


def test_warning_is_explicit():
    assert "Prototype" in PROTOTYPE_WARNING
    assert "diagnostic" in PROTOTYPE_WARNING.lower()


def test_guidelines_mention_contradiction():
    assert "contredisent" in GUIDELINES_PNEUMONIA.lower()


def test_examples_cover_three_routes():
    assert {e["route"] for e in EXAMPLES} == {"image", "text", "both"}
