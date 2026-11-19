"""Rule-based fusion + guardrails.

Pure functions: no LLM involved. This is the only place where a final label or
a refusal is produced. Thresholds are PLACEHOLDERS to be fitted on the
validation split (see README, section "Calibration").
"""
from .calibration import load_config
from .prompts import WARNING

_cfg = load_config()
TAU_HIGH = _cfg["tau_high"]      # fitted on the Kermany validation half
TAU_LOW = _cfg["tau_low"]        # (placeholders if no calibration_config.json)


def _normalise_vision(classify_result, explain_result):
    v = dict(classify_result or {})
    if "error" in v:
        return None
    label = v.get("label", "normal")
    # The tool returns a CALIBRATED confidence (temperature-scaled) already.
    # TAU_LOW/TAU_HIGH were fitted on that calibrated score, so we must NOT
    # re-derive it from the raw probability.
    score = v.get("score")
    if score is None:
        p_cal = v.get("p_pneumonia_calibrated", v.get("p_pneumonia", 0.5))
        score = p_cal if label == "pneumonie" else 1.0 - p_cal
    v["score"] = float(score)
    if explain_result and "error" not in explain_result:
        v["inside_ratio"] = explain_result.get("inside_ratio")
        v["heatmap"] = explain_result.get("heatmap")
    return v


def _normalise_nlp(result):
    n = dict(result or {})
    if "error" in n:
        return None
    return n


def fuse(vision, text, tau_high=TAU_HIGH, tau_low=TAU_LOW):
    """Return the fusion decision. vision/text may be None when unavailable."""
    v = _normalise_vision(vision, None) if vision else None
    t = _normalise_nlp(text) if text else None

    img_pos = bool(v and v.get("label") == "pneumonie")
    img_neg = bool(v and v.get("label") == "normal")
    img_ok = bool(v and v.get("score", 0.0) >= tau_low)
    img_strong = bool(v and v.get("score", 0.0) >= tau_high)

    txt_pos = bool(t and t.get("suffisant")
                   and t.get("orientation") == "infection respiratoire probable")
    txt_neg = bool(t and t.get("suffisant")
                   and t.get("orientation") == "non infectieux")
    txt_ok = bool(t and t.get("suffisant"))

    def out(coherence, final, reason, confidence, refused):
        return {"coherence": coherence, "final": final, "reason": reason,
                "confidence": confidence, "refused": refused, "warning": WARNING}

    if v is None and t is None:
        return out("insuffisant", None, "aucune modalite exploitable", "aucune", True)

    # ---- text only ----
    if v is None:
        if txt_pos:
            return out("texte_seule", "pneumonie_suspectee",
                       "note clinique compatible, pas d'image", "faible", False)
        if txt_neg:
            return out("texte_seule", "normal_suspecte",
                       "note clinique sans signe respiratoire, pas d'image", "faible", False)
        return out("insuffisant", None, "note insuffisante", "aucune", True)

    # ---- image only (or note insufficient -> fall back to the image) ----
    if t is None or not txt_ok:
        reason = "image seule" if t is None else "note insuffisante, image seule"
        if not img_ok:
            return out("insuffisant", None, f"{reason}, image incertaine", "aucune", True)
        conf = "haute" if img_strong else "moderee"
        return out("image_seule", f"{v['label']}_suspecte", reason, conf, False)

    # ---- both modalities ----
    # The image must be TRUSTED before we claim the modalities disagree:
    # below TAU_LOW we do not assert a contradiction, we assert insufficiency.
    if not img_ok:
        return out("insuffisant", None,
                   "image incertaine (score < TAU_LOW), aucun verdict bimodal",
                   "aucune", True)

    if (img_pos and txt_neg) or (img_neg and txt_pos):
        return out("contradictoire", None,
                   "image et note clinique se contredisent, refus de conclure",
                   "aucune", True)

    # concordant
    conf = "haute" if img_strong else "moderee"
    return out("concordant", f"{v['label']}_suspecte",
               "image et note clinique concordent", conf, False)
