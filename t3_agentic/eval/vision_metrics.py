"""Headline vision metrics on the official Kermany test split (624 images).

Reproducible: reads the saved per-image probabilities
(outputs/preds_kermany_full.npz) and the fitted operating point
(outputs/calibration_config.json) -- no retraining, no inference.

Reports BOTH operating points side by side:
  - the fitted threshold (fitted on the Kermany validation half, seed 42);
  - the un-fitted default 0.5, which is what the model used to do and which
    called 73.6% of the test set "pneumonia" (prevalence 62.5%).

Usage: python t3_agentic/eval/vision_metrics.py
"""
import json
import os
import sys

import numpy as np

ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

from sklearn.metrics import (accuracy_score, average_precision_score,
                             confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score)

from t3_agentic.agent.calibration import apply_temperature, load_config
from t3_agentic.eval.calibration import ece

PREDS = os.path.join(ROOT, "outputs", "preds_kermany_clean_split.npz")
LEGACY = os.path.join(ROOT, "outputs", "preds_kermany_full.npz")
OUT = os.path.join(ROOT, "outputs")


def block(p_cal, y, threshold, label):
    pred = (p_cal >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    n = len(y)
    return {
        "split": label,
        "threshold": round(float(threshold), 4),
        "n": int(n),
        "prevalence_pneumonia": round(float(y.mean()), 4),
        "predicted_positive_rate": round(float(pred.mean()), 4),
        "accuracy": round(float(accuracy_score(y, pred)), 4),
        "balanced_accuracy": round(float(0.5 * (recall_score(y, pred, pos_label=1)
                                               + recall_score(y, pred, pos_label=0))), 4),
        "auc": round(float(roc_auc_score(y, p_cal)), 4),
        "pr_auc": round(float(average_precision_score(y, p_cal)), 4),
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
        "recall_normal": round(float(recall_score(y, pred, pos_label=0)), 4),
        "recall_pneumonia": round(float(recall_score(y, pred, pos_label=1)), 4),
        "precision_normal": round(float(precision_score(y, pred, pos_label=0,
                                                       zero_division=0)), 4),
        "precision_pneumonia": round(float(precision_score(y, pred, pos_label=1,
                                                          zero_division=0)), 4),
        "f1_normal": round(float(f1_score(y, pred, pos_label=0)), 4),
        "f1_pneumonia": round(float(f1_score(y, pred, pos_label=1)), 4),
        "f1_macro": round(float(f1_score(y, pred, average="macro")), 4),
        "ece": round(ece(p_cal, y, threshold=threshold)[0], 4),
    }


def main():
    cfg = load_config()
    t = cfg["label_threshold"]
    if not os.path.exists(PREDS):
        print("clean split predictions missing; run eval/reproduce_kaggle_split.py")
        return
    d = np.load(PREDS, allow_pickle=True)
    split = d["split"]
    s_test = split == "test"
    s_val = split == "val"
    p, y = d["p"][s_test], d["y"][s_test]
    p_cal = np.array([apply_temperature(x, cfg["temperature"]) for x in p])

    report = {
        "split": "Kermany clean patient-disjoint test slice (940, rebuilt from "
                 "the Kaggle split and verified against the Kaggle run log)",
        "operating_point": "fitted threshold (Kermany validation half, seed 42)",
        "accuracy": None, "auc": None, "confusion_matrix": None,
        "recall_normal": None, "recall_pneumonia": None,
        "precision_normal": None, "precision_pneumonia": None,
        "f1_normal": None, "f1_pneumonia": None, "f1_macro": None,
        "n_normal": int((y == 0).sum()), "n_pneumonia": int((y == 1).sum()),
    }
    main_block = block(p_cal, y, t, "Kermany clean patient-disjoint test slice (940)")
    for k in list(report):
        if report[k] is None:
            report[k] = main_block[k]

    report["detail_fitted_threshold"] = main_block
    report["detail_default_0.5"] = block(
        p_cal, y, 0.5, "Kermany clean test slice (940) at default 0.5")
    report["note"] = ("the default 0.5 boundary over-called pneumonia "
                      "(73.6% predicted positive vs 62.5% prevalence); the "
                      "fitted threshold fixes the operating point")

    # what actually changes between the two boundaries
    old = (p > 0.5).astype(int)
    new = (p_cal >= t).astype(int)
    to_normal = np.where((old == 1) & (new == 0))[0]
    to_pneu = np.where((old == 0) & (new == 1))[0]
    report["decision_changes_vs_default_0.5"] = {
        "pneumonia_to_normal": int(len(to_normal)),
        "normal_to_pneumonia": int(len(to_pneu)),
        "unchanged": int((old == new).sum()),
        "pneumonia_to_normal_that_were_normal_in_truth":
            int((y[to_normal] == 0).sum()),
        "pneumonia_to_normal_that_were_pneumonia_in_truth":
            int((y[to_normal] == 1).sum()),
        "examples_pneumonia_to_normal": [
            {"name": str(d["names"][j]), "p_pneumonia": round(float(p[j]), 4),
             "p_calibrated": round(float(p_cal[j]), 4),
             "truth": "normal" if y[j] == 0 else "pneumonia"}
            for j in to_normal[:5]],
    }

    p_all = d["p"]
    y_all = d["y"]
    p_cal_val = np.array([apply_temperature(x, cfg["temperature"]) for x in p_all[s_val]])
    report["detail_validation_slice"] = block(
        p_cal_val, y_all[s_val], t, "Kermany clean validation slice (871, fitted on)")
    report["detail_official_test_folder"] = None
    if os.path.exists(LEGACY):
        leg = np.load(LEGACY)
        pl = np.array([apply_temperature(x, cfg["temperature"]) for x in leg["p"]])
        report["detail_official_test_folder"] = block(
            pl, leg["y"], t,
            "official Kermany test folder (624): the training notebook's pool "
            "contained this folder, and it is a harder slice than the pool's "
            "own patient-disjoint split")

    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, "vision_metrics.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


if __name__ == "__main__":
    main()
