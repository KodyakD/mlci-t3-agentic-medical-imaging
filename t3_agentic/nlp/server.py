"""MCP NLP server: analyze_text."""
import os
import sys

from mcp.server.fastmcp import FastMCP

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scrub import scrub
from terms import find_symptoms, orientation

mcp = FastMCP("nlp")


@mcp.tool()
def analyze_text(note: str) -> dict:
    """Analyse une note clinique : entites symptomes, orientation, suffisance.

    Les mentions diagnostiques explicites sont retirees avant analyse.
    """
    if not note or not note.strip():
        return {"entities": [], "denied": [], "orientation": "note vide",
                "suffisant": False, "evidence": [], "redactions": 0}
    cleaned, redactions = scrub(note)
    symptoms, denied = find_symptoms(cleaned)
    orient, suffisant = orientation(symptoms, denied, cleaned)
    return {
        "entities": symptoms,
        "denied": denied,
        "orientation": orient,
        "suffisant": bool(suffisant),
        "evidence": symptoms,
        "redactions": redactions,
    }


if __name__ == "__main__":
    mcp.run(transport="stdio")
