"""The operating point: one decision rule, fitted on the validation half.

The model used to decide "pneumonia" at a hardcoded 0.5 on the RAW
probability, while the trust thresholds were fitted on the CALIBRATED score.
At 0.5 the model called 73.6% of the official Kermany test set "pneumonia"
(prevalence 62.5%) and put recall on NORMAL at 0.70 -- the complaint that the
model "guesses everything as pneumonia". These tests pin the fix.
"""
import json
import os
import sys

import numpy as np

ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

from t3_agentic.agent.calibration import (LABEL_THRESHOLD_DEFAULT, decide,
                                          load_config)
from t3_agentic.eval.calibration import ece
from t3_agentic.eval.fit_thresholds import fit_label_threshold

CONFIG = os.path.join(ROOT, "outputs", "calibration_config.json")


def test_decide_uses_the_fitted_threshold_not_0_5():
    cfg = {"temperature": 1.0, "label_threshold": 0.9}
    label, p_cal, score = decide(0.85, cfg)
    assert label == "normal" and abs(p_cal - 0.85) < 1e-9
    assert abs(score - 0.15) < 1e-9
    label, _, score = decide(0.95, cfg)
    assert label == "pneumonie" and abs(score - 0.95) < 1e-9


def test_decide_applies_the_temperature_before_deciding():
    # a raw 0.99 with temperature 2 -> calibrated 0.90 < 0.9? no: 0.9 -> label
    cfg = {"temperature": 2.0, "label_threshold": 0.9}
    label, p_cal, _ = decide(0.99, cfg)
    assert abs(p_cal - 0.9087) < 1e-3   # sigmoid(logit(0.99)/2)
    assert label == "pneumonie"


def test_shipped_config_has_a_fitted_label_threshold():
    if not os.path.exists(CONFIG):
        return  # no fitted config in this checkout: nothing to assert
    with open(CONFIG, encoding="utf-8") as f:
        cfg = json.load(f)
    assert "label_threshold" in cfg
    assert 0.5 < float(cfg["label_threshold"]) < 1.0, \
        "the fitted threshold must be above the over-calling 0.5 default"
    assert float(cfg["label_threshold"]) != LABEL_THRESHOLD_DEFAULT


def test_load_config_exposes_the_label_threshold():
    cfg = load_config()
    assert "label_threshold" in cfg
    assert 0.0 < cfg["label_threshold"] < 1.0


def test_fit_label_threshold_beats_0_5_on_a_shifted_prior():
    # a model whose scores are pushed towards the positive class
    rng = np.random.default_rng(0)
    y = np.array([0] * 40 + [1] * 60)
    p = np.clip(y * 0.9 + rng.normal(0, 0.12, len(y)), 0.001, 0.999)

    def bal_acc(th):
        pred = (p >= th).astype(int)
        tpr = ((pred == 1) & (y == 1)).sum() / max((y == 1).sum(), 1)
        tnr = ((pred == 0) & (y == 0)).sum() / max((y == 0).sum(), 1)
        return 0.5 * (tpr + tnr)

    t = fit_label_threshold(p, y)
    assert bal_acc(t) >= bal_acc(0.5)
    assert t > 0.5


def test_ece_follows_the_threshold():
    p = np.array([0.55, 0.95, 0.10, 0.99, 0.20, 0.80])
    y = np.array([0, 1, 0, 1, 0, 1])
    e_low, rows_low = ece(p, y, threshold=0.5)
    e_high, rows_high = ece(p, y, threshold=0.9)
    assert 0.0 <= e_low <= 1.0 and 0.0 <= e_high <= 1.0
    # the confidence of a sample depends on the decision boundary
    assert rows_low != rows_high
