"""Critic pass: a second LLM reads the drafted answer against the fusion facts.

The critic may only REJECT or REVISE an answer; it can never introduce a new
medical claim, because the revised text must pass the same provenance check as
the original (numbers must come from the fusion/vision/text JSON) and the
mandatory warning is re-appended in code.

Verdicts:
  ok      -> the answer is kept as is
  revise  -> the critic returns a corrected answer (still provenance-checked)
  reject  -> the deterministic template answer is used instead
"""
import re

from .prompts import WARNING

CRITIC_SYSTEM = """Tu es un critique d'un prototype d'analyse d'imagerie medicale.
Tu recois la question, les FAITS (JSON) et une REPONSE PROPOSEE.
Verifie strictement, dans cet ordre:
1. La reponse affirme-t-elle un diagnostic, une certitude ("c'est une pneumonie",
   "diagnostic de pneumonie")? Si oui: REJECT.
2. La reponse contredit-elle les FAITS (par exemple: les faits indiquent une
   fusion faible ou un refus, mais la reponse affirme une suspicion forte)? Si
   oui: REJECT.
3. La reponse invente-t-elle un nombre ou un resultat absent des FAITS? Si oui: REJECT.
4. La reponse est-elle simplement malformee, trop longue, ou contient-elle des
  Details non necessaires? Si oui: REVISE.

Reponds en JSON strict: {"verdict": "ok|revise|reject", "issues": ["..."], "revised_answer": "..."}
(revised_answer vide si verdict != "revise")"""

# Deterministic patterns the critic must not let through, whatever it says.
OVERCLAIM = re.compile(
    r"(diagnostic (de|d')\s*pneumonie|il s'agit d'une? pneumonie|"
    r"pneumonie (confirm\w*|constat\w*|diagnostiqu\w*)|"
    r"pr[ée]sence d'une pneumonie|"
    r"confiance (élev\w+|fort\w*|haute))", re.I)

# "the fusion refused" - only true when the fusion really refused.
REFUSAL_CLAIM = re.compile(
    r"(refuse de conclure|ne permet pas de conclure|refus de conclure|"
    r"fusion refuse|a refuse de conclure)", re.I)
# "the fusion concluded a suspicion" - only true when it did.
SUSPICION_CLAIM = re.compile(
    r"(suspicion de pneumonie|conclu (avec|à) une suspicion|"
    r"orient(e|e) vers une suspicion)", re.I)
# an internal JSON key quoted in prose (p_pneumonia_calibrated, inside_ratio, ...)
KEY_LEAK = re.compile(r"\b([a-z][a-z0-9]*(?:_[a-z0-9]+)+)\b")


def contradiction_with_fusion(text, fusion):
    """Return a contradiction reason, or None.

    A critic revision must not claim something the fusion did not produce:
    saying "the fusion refused" when it did not is exactly the kind of
    confident-sounding lie the critic is supposed to prevent.
    """
    if not fusion:
        return None
    refused = bool(fusion.get("refused"))
    if REFUSAL_CLAIM.search(text or "") and not refused:
        return "claims a refusal the fusion did not produce"
    if SUSPICION_CLAIM.search(text or "") and refused:
        return "claims a suspicion the fusion refused to make"
    if refused:
        hedges = re.search(r"(incertitude|ind[ée]termin|faible|prudence|avis humain|"
                           r"ne permet pas|sans certitude|exploratoire)", text or "")
        if not REFUSAL_CLAIM.search(text or "") and not hedges:
            return "does not acknowledge the refusal of the fusion"
    return None


def key_leaks(text, facts):
    """JSON keys quoted in the prose that do not exist in the facts."""
    allowed = set()
    try:
        def walk(o):
            if isinstance(o, dict):
                for k, v in o.items():
                    allowed.add(str(k))
                    walk(v)
            elif isinstance(o, (list, tuple)):
                for v in o:
                    walk(v)
        walk(facts)
    except Exception:
        pass
    allowed |= {"prototype", "pneumonie", "normal", "fusion", "image", "texte"}
    return [k for k in set(KEY_LEAK.findall(text or "")) if k not in allowed]


def _numbers(text):
    return set(re.findall(r"\d+(?:[.,]\d+)?", text or ""))


def parse_critic_reply(content):
    """Extract {verdict, issues, revised_answer} from a (possibly noisy) reply."""
    import json
    if not content:
        return {"verdict": "ok", "issues": [], "revised_answer": ""}
    m = re.search(r"\{.*\}", content, re.S)
    raw = m.group(0) if m else content
    try:
        obj = json.loads(raw)
    except Exception:
        # the model sometimes wraps the JSON in prose: pull the fields by regex
        vm = re.search(r'"verdict"\s*:\s*"(\w+)"', content)
        rm = re.search(r'"revised_answer"\s*:\s*"(.*?)"\s*[,}]', content, re.S)
        obj = {"verdict": vm.group(1) if vm else "ok",
               "issues": [],
               "revised_answer": rm.group(1) if rm else ""}
    verdict = str(obj.get("verdict", "ok")).lower()
    if verdict not in ("ok", "revise", "reject"):
        verdict = "ok"
    issues = obj.get("issues") or []
    if isinstance(issues, str):
        issues = [issues]
    return {"verdict": verdict,
            "issues": [str(i)[:200] for i in issues][:5],
            "revised_answer": str(obj.get("revised_answer") or "").strip()}


def critique(answer, facts, question, ollama_chat=None):
    """Run the critic. Returns (verdict, revised_answer, issues).

    The critic is advisory: if the LLM is unavailable we keep the answer.
    """
    import json as _json
    if ollama_chat is None:
        from .agent import ollama_chat  # late import avoids a cycle
    payload = _json.dumps(facts, ensure_ascii=False)
    try:
        msg = ollama_chat([
            {"role": "system", "content": CRITIC_SYSTEM},
            {"role": "user", "content": f"QUESTION: {question}\nFAITS: {payload}\n"
                                        f"REPONSE PROPOSEE: {answer}"},
        ])
        content = (msg.get("message", {}) or {}).get("content", "")
    except Exception:
        return "ok", "", ["critic unavailable"]

    out = parse_critic_reply(content)
    verdict, issues, revised = out["verdict"], out["issues"], out["revised_answer"]

    # deterministic veto: the critic cannot talk us into keeping an overclaim
    if OVERCLAIM.search(revised or answer or ""):
        verdict = "reject"
        issues = issues + ["overclaim detected by the deterministic guard"]

    if verdict == "revise" and not revised:
        verdict = "ok"
    return verdict, revised, issues


def apply_critic(verdict, revised, original, template_answer, facts):
    """Return (final_answer, used_template).

    A revision is accepted only if it passes the same provenance check as the
    original answer and keeps the mandatory warning.
    """
    from .agent import _provenance_ok
    if verdict == "reject":
        return template_answer, True
    if verdict == "revise" and revised:
        allowed = _json_dumps(facts)
        if not _provenance_ok(revised, allowed):
            # a revision that invented a number is dropped: keep the original
            return original, False
        contra = contradiction_with_fusion(revised, facts.get("fusion") if isinstance(facts, dict) else None)
        if contra:
            # the critic contradicted the fusion: fall back to the template
            return template_answer, True
        leaks = key_leaks(revised, facts)
        if leaks:
            # internal JSON keys leaked into the prose: keep the original
            return original, False
        stripped = re.sub(r"[^.]*prototype\s+p[ée]dagogique[^.]*\.", "", revised,
                          flags=re.I).strip()
        return (stripped + " " + WARNING).strip(), False
    return original, False


def _json_dumps(obj):
    import json
    return json.dumps(obj, ensure_ascii=False)
