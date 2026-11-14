"""MCP resources/prompts server (resources, not only tools)."""
import os
import sys

from mcp.server.fastmcp import FastMCP

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from content import EXAMPLES, GUIDELINES_PNEUMONIA, PROTOTYPE_WARNING
from retrieval import search

mcp = FastMCP("resources")


@mcp.tool()
def get_guideline(topic: str = "") -> str:
    """Repères de lecture d'une radiographie thoracique (ressource guidelines://pneumonia)."""
    return GUIDELINES_PNEUMONIA


@mcp.tool()
def search_guidelines(query: str, k: int = 3) -> list[dict]:
    """Recherche les passages de reference les plus pertinents pour une question.

    Utilise pour les questions de connaissances (radiographie, signes, limites,
    ethique) quand aucune image ni note n'est fournie.
    """
    return search(query, k=k)


@mcp.resource("guidelines://pneumonia")
def pneumonia_guidelines() -> str:
    """Repères de lecture d'une radiographie thoracique."""
    return GUIDELINES_PNEUMONIA


@mcp.resource("ethics://prototype-warning")
def prototype_warning() -> str:
    """Avertissement non diagnostique obligatoire."""
    return PROTOTYPE_WARNING


@mcp.resource("examples://demo-cases")
def demo_cases() -> str:
    """Exemples de requetes et de routage attendu."""
    return str(EXAMPLES)


@mcp.prompt()
def cautious_radiology_assistant(question: str) -> list[dict]:
    """Systeme prompt pour un assistant d'interpretation prudent."""
    return [
        {"role": "user",
         "content": "Tu es un assistant prudent d'interpretation d'imagerie medicale. "
                    "Aucun diagnostic. Si l'image et le texte se contredisent, "
                    "refuse de conclure. Termine toujours par: "
                    "Prototype pedagogique, ne constitue pas un diagnostic medical.\n\n"
                    f"{question}"},
    ]


if __name__ == "__main__":
    mcp.run(transport="stdio")
