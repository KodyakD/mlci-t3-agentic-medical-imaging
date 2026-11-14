"""Remove explicit diagnostic mentions before NLP analysis.

The TP requires an explicit cleaning step: a note that already contains the
diagnosis must not leak it into the agent's reasoning.
"""
import re

DIAG_PATTERNS = [
    r"\b(?:pas de |pas d['’]|suspicion de |presence de |diagnostic de |atteint de )?"
    r"(?:pneumonie|pneumonia)\b",
    r"\b(?:tuberculose|tuberculosis|tuberculose pulmonaire|tb)\b",
    r"\b(?:copd|bpco|asthme|asthma)\b",
    r"\b(?:cancer|tumeur|tumeur maligne|tumour|tumor|malignite|malignancy)\b",
    r"\b(?:insuffisance cardiaque|cardiac failure|heart failure)\b",
    r"\b(?:embolie pulmonaire|pulmonary embolism)\b",
    r"\b(?:sarcoidose|tuberculose miliaire)\b",
]

REDACTION = "[DIAG_REDACTED]"


def scrub(note):
    """Return (cleaned_note, redactions)."""
    out = note
    count = 0
    for pat in DIAG_PATTERNS:
        out, n = re.subn(pat, REDACTION, out, flags=re.I)
        count += n
    return out, count
