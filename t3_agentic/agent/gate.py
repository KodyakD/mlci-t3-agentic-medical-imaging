"""Deterministic modality gate and tool-choice validation.

The PDF requires that a text-only question never triggers the CNN. That is
enforced here, in code, not in the prompt.
"""
import re

IMAGE_TOOLS = ("classify_image", "explain_image")
TEXT_TOOLS = ("analyze_text",)
RESOURCE_TOOLS = ("get_guideline",)


def allowed_tools(has_image, has_note):
    """Tools the agent may call, given what the user actually provided.

    - image only  -> vision tools
    - note only   -> NLP tool
    - neither     -> the MCP resource (knowledge question), no CNN, no NLP
    """
    tools = set()
    if has_image:
        tools.update(IMAGE_TOOLS)
    if has_note:
        tools.update(TEXT_TOOLS)
    if not tools:
        tools.update(RESOURCE_TOOLS)
    return tools


def required_tools(has_image, has_note):
    """Tools the harness will run regardless of what the LLM proposed."""
    req = set()
    if has_image:
        req.update(IMAGE_TOOLS)
    if has_note:
        req.update(TEXT_TOOLS)
    return req


def validate_tool_choice(proposed, allowed):
    """Keep only known tools that are allowed; preserve order; drop duplicates."""
    if not proposed:
        return []
    out = []
    for name in proposed:
        if name in allowed and name not in out:
            out.append(name)
    return out


def complete_tools(called, required):
    """Harness completion: missing required tools are appended (logged as forced)."""
    missing = [t for t in ("classify_image", "explain_image", "analyze_text")
               if t in required and t not in called]
    return missing


def fallback_route(user_msg, has_image, has_note):
    """Deterministic router used when the LLM proposes nothing usable."""
    q = (user_msg or "").lower()
    if has_image and has_note:
        return ["classify_image", "explain_image", "analyze_text"]
    if has_image:
        return ["classify_image", "explain_image"]
    if has_note:
        return ["analyze_text"]
    # knowledge question (no image, no note) -> read the MCP resource
    return ["get_guideline"]
