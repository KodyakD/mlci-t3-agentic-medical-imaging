"""Optional phrasing bank for the deterministic report (no LLM).

When T3_PHRASING_BANK=1, the qualitative, number-free sentences of the answer
are sampled from built-in paraphrase pools. Numeric facts (probabilities,
confidence, Grad-CAM ratio) are always injected by code and never come from this
bank, and the mandatory non-diagnostic clause is always preserved. Disabled by
default so the canonical wording stays stable.
"""
import os
import random

ENV = "T3_PHRASING_BANK"
CLAUSE = "ne constitue pas un diagnostic medical"
_ENABLED = os.environ.get(ENV, "").strip().lower() not in ("", "0", "false", "no")
_rng = random.Random(os.environ.get("T3_PHRASING_SEED") or None)

POOLS = {
    "OPENING": [
        "Classe favorisee par le modele : {label}.",
        "Le classifieur oriente vers : {label}.",
        "Prediction de l'image : {label}.",
        "Resultat de la classification : {label}.",
    ],
    "GRADCAM_NOTE": [
        "Cette carte montre des zones influentes pour le modele; elle ne localise ni ne confirme une lesion.",
        "La carte indique les regions qui ont pese sur la decision; elle ne prouve ni ne situe une lesion.",
        "Il s'agit d'une carte d'attention du modele, non d'une localisation ni d'une confirmation de lesion.",
    ],
    "NONOTE": [
        "Symptomes cliniques non renseignes : l'image ne permet pas de les determiner. Ajoutez une note clinique pour analyser les symptomes, s'il y en a.",
        "Aucune note clinique fournie : le modele ne peut pas etablir de symptomes a partir de l'image. Ajoutez une note pour les analyser.",
    ],
    "NORMAL_SYM": [
        "Aucun symptome n'a ete indique dans une note. Une prediction normale sur l'image ne signifie pas que le patient est sans symptomes.",
    ],
    "FUSION": [
        "Fusion : {coh} ; resultat de triage experimental : {tri}.",
        "Decision de fusion : {coh} ; triage experimental : {tri}.",
        "Resultat de fusion : {coh} ; niveau de triage : {tri}.",
    ],
    "DISCLAIMER": [
        "Prototype pedagogique, ne constitue pas un diagnostic medical.",
        "Rappel : ce prototype pedagogique ne constitue pas un diagnostic medical.",
        "Avertissement : outil pedagogique, ne constitue pas un diagnostic medical.",
    ],
}

def pick(key, default, **fmt):
    """Return a sampled variant when enabled, else the canonical default."""
    pool = POOLS.get(key, [])
    if _ENABLED and pool:
        tpl = _rng.choice(pool)
        try:
            return tpl.format(**fmt)
        except (KeyError, IndexError):
            return default
    return default

def disclaimer(default):
    """A varied closing that always keeps the mandatory non-diagnostic clause."""
    out = pick("DISCLAIMER", default)
    return out if CLAUSE in out else default
