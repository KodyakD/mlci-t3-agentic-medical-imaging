"""Tests for the rule-based fusion (pure functions, no LLM)."""
import os
import sys

sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..")))

from t3_agentic.agent.fusion import fuse


def vision(label="pneumonie", score=0.95, suspect=False):
    """score = confidence in the PREDICTED label; p_pneumonia is derived."""
    p = score if label == "pneumonie" else 1.0 - score
    return {"label": label, "p_pneumonia": p, "score": score,
            "suspect": suspect}


def text(orientation="infection respiratoire probable", suffisant=True):
    return {"orientation": orientation, "suffisant": suffisant, "entities": []}


def test_concordant_positive():
    # score above the fitted TAU_HIGH -> "haute" confidence
    r = fuse(vision("pneumonie", 0.9999), text())
    assert r["coherence"] == "concordant"
    assert r["final"] == "pneumonie_suspecte"
    assert r["confidence"] == "haute"
    assert r["refused"] is False


def test_concordant_but_below_tau_high_is_moderate():
    r = fuse(vision("pneumonie", 0.90), text())
    assert r["confidence"] == "moderee"


def test_fusion_uses_the_fitted_thresholds():
    from t3_agentic.agent.calibration import load_config
    from t3_agentic.agent.fusion import TAU_LOW, TAU_HIGH
    cfg = load_config()
    assert TAU_LOW == cfg["tau_low"] and TAU_HIGH == cfg["tau_high"]


def test_concordant_negative():
    r = fuse(vision("normal", 0.95), text("non infectieux", True))
    assert r["coherence"] == "concordant"
    assert r["final"] == "normal_suspecte"


def test_contradiction_refuses():
    r = fuse(vision("pneumonie", 0.95), text("non infectieux", True))
    assert r["coherence"] == "contradictoire"
    assert r["final"] is None
    assert r["refused"] is True


def test_contradiction_other_direction_refuses():
    r = fuse(vision("normal", 0.9), text("infection respiratoire probable", True))
    assert r["coherence"] == "contradictoire"
    assert r["refused"] is True


def test_insufficient_note_falls_back_to_image_only():
    r = fuse(vision("pneumonie", 0.95), text("orientation insuffisante", False))
    assert r["coherence"] == "image_seule"
    assert r["final"] == "pneumonie_suspecte"


def test_uncertain_image_refuses():
    r = fuse(vision("pneumonie", 0.40), None)
    assert r["coherence"] == "insuffisant"
    assert r["refused"] is True


def test_suspect_gradcam_downgrades_confidence():
    r = fuse(vision("pneumonie", 0.95, suspect=True), None)
    assert r["confidence"] == "faible"


def test_text_only_never_high_confidence():
    r = fuse(None, text())
    assert r["coherence"] == "texte_seule"
    assert r["confidence"] == "faible"


def test_warning_always_present():
    for args in [(vision(), text()), (None, None), (vision("normal", 0.3), None)]:
        assert "Prototype" in fuse(*args)["warning"]


def test_untrusted_image_does_not_produce_a_contradiction():
    # image below TAU_LOW + a note denying infection: we must NOT claim the
    # modalities disagree (the image is not trusted) -> insufficiency
    r = fuse(vision("pneumonie", 0.40), text("non infectieux", True))
    assert r["coherence"] == "insuffisant"
    assert r["refused"] is True


def test_trusted_contradiction_still_refuses():
    r = fuse(vision("pneumonie", 0.99), text("non infectieux", True))
    assert r["coherence"] == "contradictoire"
    assert r["refused"] is True
