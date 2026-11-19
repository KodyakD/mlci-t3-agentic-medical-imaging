"""Symptom lexicon (French + English) with negation handling.

Negation matters: "pas de fievre" must not count as fever, and a note that
explicitly denies respiratory symptoms is informative (it can contradict the
image, which is exactly the refusal path the TP asks for).
"""
import re
import unicodedata


def strip_accents(text):
    """NFKD-decompose and drop combining marks so accented French
    ('fièvre', 'râles', 'dyspnée') matches the unaccented lexicon."""
    if not text:
        return text or ""
    norm = unicodedata.normalize("NFKD", text)
    return "".join(c for c in norm if not unicodedata.combining(c))


SYMPTOMS = [
    # French nouns AND verb inflections ("tousse", "tousser", "expectore", ...)
    (r"\btoux\w*\b|\btousse\w*\b|\btoussote\w*|\btouffe\b|\bquintes? de toux\b|"
     r"\bcough(?:s|ing|ed)?\b", "toux"),
    (r"\bfievre\b|\bfebril\w*|\bhyperthermie\b|\bfever(?:s|ish)?\b", "fievre"),
    (r"\bdouleurs? (?:thoraciques?|(?:dans |a la |de la )?poitrine)\b|"
     r"\bmaux? de poitrine\b|\boppression thoracique\b|\bprecordialgie\w*\b|"
     r"\bthoracalgie\w*\b|\bchest pain\b|\bchest discomfort\b", "douleur thoracique"),
    (r"\bdyspn\w*e\b|\bdyspn(?:ea|oea)\b|\bessouffl\w*|\bmanque d'air\b|"
     r"\bshort(?:ness)? of breath\b|\bbreathlessness\b|"
     r"\brespiration (?:rapide|courte|difficile|laborieuse)\b|"
     r"\brespire (?:rapidement|mal|difficilement)\b|"
     r"\btachypn\w*|\bpolypn\w*|\borthopn\w*|\btirage\b", "dyspnee"),
    (r"\bexpector\w*|\bsputum\b|\bphlegm\b|\bcrachats?\b|\bcrache\w*", "expectoration"),
    (r"\bfrissons?\b|\bfrissonne\w*|\bgrelott\w*|\bshiver\w*|\bchills?\b", "frissons"),
    (r"\bfatigue\b|\bfatig\w*|\bmalaise\b|\basth[ée]nie\b|\badynamie\b|"
     r"\bepuisement\b|\btiredness\b", "fatigue"),
    (r"\bcrepitants?\b|\bcrepit\w*|\brales?\b|\bcrackles?\b|\bcrackl\w*|"
     r"\bwheez\w*|\bsibilants?\b|\bsibilances\b|\bronchis?\b|\brhonch\w*", "rales"),
]

STRONG = {"toux", "fievre", "dyspnee", "douleur thoracique", "expectoration"}

NEGATIONS = [
    r"\bpas de\b", r"\bpas d['’]", r"\baucun(?:e)?\b", r"\bsans\b",
    r"\bne (?:pas|presente)\b", r"\bn['’]a pas\b", r"\babsence de\b",
    r"\bjamais\b", r"\bno\b", r"\bwithout\b", r"\bdenies?\b", r"\bdenied\b",
    r"\bnot\b", r"\bnegative for\b", r"\bno evidence of\b", r"\bfree of\b",
]

RESPIRATORY_DENIAL = re.compile(
    r"asymptomatique|"
    r"(pas de|pas d['’]|aucun|sans|absence de|n['’]a pas de|no|without|negat\w*)\s+"
    r"(signe?s?\s+)?(toux|cough|fievre|fever|dyspn\w*|shortness of breath|expectoration|sputum|"
    r"douleur thoracique|chest pain|symptom\w*\s+respirat\w*|infectieux|respiratoire)",
    re.I)

SYMPTOM_DENIAL = re.compile(
    r"(pas de|pas d['’]|aucun|sans|n['’]a pas de|absence de|no|without|not)\s+"
    r"(toux|cough|fievre|fever|dyspn\w*|expectoration|sputum|douleur thoracique|"
    r"chest pain|frissons|chills|fatigue)",
    re.I)

# A negation can also follow the symptom ("toux absente", "fievre negative").
POST_NEGATION = re.compile(
    r"\b(absente|absent|abolie|aboli|negative|negatif|innexistant|absorb)\w*\b"
    r"|\bnon (?:retrouve|objective|present)\w*", re.I)


CLAUSE_BREAK = re.compile(
    r"(?:,|;|\.|\bavec\b|\bet\b|\bmais\b|\bpuis\b|\bcependant\b|however|\bbut\b|"
    r"\band\b|\bwith\b|\bwhile\b)", re.I)


def _negated(text, start):
    """True if a negation governs the symptom at `start`.

    The negation scope stops at the previous clause break, otherwise
    "sans toux, avec fievre" would negate the fever too. A negation may also
    FOLLOW the symptom ("toux absente"), and the French "ne <verbe> pas" form
    is detected on both sides.
    """
    window = text[max(0, start - 60):start]
    breaks = list(CLAUSE_BREAK.finditer(window))
    if breaks:
        window = window[breaks[-1].end():]
    if any(re.search(p, window) for p in NEGATIONS):
        return True
    after = text[start:start + 30]
    if POST_NEGATION.search(after):
        return True
    # "ne <verbe> pas" : "ne" just before the symptom, "pas" just after the word
    if re.search(r"\bne\s*$", window):
        tail = after
        word = re.match(r"\S+", tail)
        if word:
            tail = tail[word.end():]
        if re.match(r"\s*(ne\s+\w+\s+)?pas\b", tail):
            return True
    return False


def find_symptoms(text):
    """Return (symptoms_found, symptoms_denied)."""
    found, denied = set(), set()
    low = strip_accents(text.lower())
    for pat, label in SYMPTOMS:
        for m in re.finditer(pat, low):
            if _negated(low, m.start()):
                denied.add(label)
            else:
                found.add(label)
    return sorted(found - denied), sorted(denied & found or denied)


def orientation(symptoms, denied, raw_text):
    """Map symptoms to a clinical orientation.

    Rule: 'infection respiratoire probable' needs at least 2 distinct symptoms
    AND at least 1 of them strong (fever, dyspnoea, chest pain, cough with
    sputum...). Weaker-only combinations stay 'orientation insuffisante'.
    """
    cleaned = strip_accents(raw_text)
    strong = [s for s in symptoms if s in STRONG]
    if len(symptoms) >= 2 and len(strong) >= 1:
        return "infection respiratoire probable", True
    if symptoms:
        # symptoms are present: a denial elsewhere does NOT make it non-infectious
        return "orientation insuffisante", False
    if RESPIRATORY_DENIAL.search(cleaned):
        return "non infectieux", True
    return "aucun symptome respiratoire detecte", False
