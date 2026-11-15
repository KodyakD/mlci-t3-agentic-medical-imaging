"""Phase 1.4 : bootstrap 95% confidence intervals on the saved predictions.

No retraining: we resample the saved per-image probabilities.
Reported for the TEST half of the official Kermany test (312 images, seed 42)
and, for reference, for the full official test (624 images).

Usage: python t3_agentic/eval/bootstrap_ci.py
"""
import json
import os
import sys

import numpy as np

ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

from sklearn.metrics import roc_auc_score

from t3_agentic.agent.calibration import apply_temperature, load_config
from t3_agentic.eval.calibration import ece

PREDS = os.path.join(ROOT, "outputs", "preds_kermany_clean_split.npz")
LEGACY = os.path.join(ROOT, "outputs", "preds_kermany_full.npz")
OUT = os.path.join(ROOT, "outputs")
N_BOOT = 1000
SEED = 42


def metrics(p, y, cfg):
    p_cal = np.array([apply_temperature(x, cfg["temperature"]) for x in p])
    t = cfg["label_threshold"]
    pred = (p_cal >= t).astype(int)
    conf = np.where(p_cal >= t, p_cal, 1 - p_cal)
    trusted = conf >= cfg["tau_low"]
    e, _ = ece(p_cal, y)
    try:
        a = roc_auc_score(y, p_cal)
    except ValueError:
        a = float("nan")
    return {
        "accuracy": float((pred == y).mean()),
        "auc": float(a),
        "ece": float(e),
        "trusted_coverage": float(trusted.mean()),
        "trusted_accuracy": float((pred[trusted] == y[trusted]).mean()) if trusted.any() else float("nan"),
        "n_trusted": int(trusted.sum()),
    }


def bootstrap(p, y, cfg, n_boot=N_BOOT):
    rng = np.random.default_rng(SEED)
    idx0 = np.where(y == 0)[0]
    idx1 = np.where(y == 1)[0]
    keys = ["accuracy", "auc", "ece", "trusted_coverage", "trusted_accuracy"]
    draws = {k: [] for k in keys}
    for _ in range(n_boot):
        take = np.concatenate([rng.choice(idx0, len(idx0), replace=True),
                               rng.choice(idx1, len(idx1), replace=True)])
        m = metrics(p[take], y[take], cfg)
        for k in keys:
            draws[k].append(m[k])
    out = {}
    for k in keys:
        v = np.array(draws[k], dtype=float)
        v = v[~np.isnan(v)]
        if len(v) == 0:
            out[k] = {"mean": None, "ci95": None}
            continue
        out[k] = {"mean": round(float(v.mean()), 4),
                  "ci95": [round(float(np.percentile(v, 2.5)), 4),
                           round(float(np.percentile(v, 97.5)), 4)],
                  "bootstrap_sd": round(float(v.std(ddof=1)), 4)}
    return out


def main():
    cfg = load_config()
    d = np.load(PREDS, allow_pickle=True)
    split = d["split"]
    s_test = split == "test"
    s_val = split == "val"
    p, y = d["p"], d["y"]
    report = {
        "method": f"{N_BOOT} resamples stratifiees (par classe), seed {SEED}, "
                   "recalcul complet a chaque echantillon",
        "calibration_used": {"tau_low": cfg["tau_low"], "tau_high": cfg["tau_high"],
                             "temperature": cfg["temperature"],
                             "label_threshold": cfg["label_threshold"],
                             "source": cfg["source"]},
        "clean_test_slice_940": {
            "n": int(s_test.sum()),
            "point_estimate": {k: round(v, 4) for k, v in
                               metrics(p[s_test], y[s_test], cfg).items()},
            "bootstrap": bootstrap(p[s_test], y[s_test], cfg),
        },
        "clean_validation_slice_871": {
            "n": int(s_val.sum()),
            "point_estimate": {k: round(v, 4) for k, v in
                               metrics(p[s_val], y[s_val], cfg).items()},
            "bootstrap": bootstrap(p[s_val], y[s_val], cfg),
        },
    }
    if os.path.exists(LEGACY):
        leg = np.load(LEGACY)
        report["official_test_folder_624"] = {
            "n": int(len(leg["p"])),
            "point_estimate": {k: round(v, 4) for k, v in
                               metrics(leg["p"], leg["y"], cfg).items()},
            "bootstrap": bootstrap(leg["p"], leg["y"], cfg),
        }
    with open(os.path.join(OUT, "bootstrap_ci.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
