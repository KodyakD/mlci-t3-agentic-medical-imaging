"""A1 + B2 + B5 : fit the label threshold, the trust thresholds and a temperature.

Protocol (disclosed in the report):
  - untouched OFFICIAL Kermany test split (624 images, never seen during training);
  - stratified 312 / 312 split, seed 42;
  - thresholds and temperature are fitted on the VALIDATION half only;
  - everything is reported on the TEST half;
  - NIH is never used for fitting (it stays the unfitted domain-shift check).

What is fitted on the validation half:
  - temperature     : minimises NLL (logit scaling);
  - label_threshold : the decision boundary on the CALIBRATED score, chosen to
                      maximise balanced accuracy (Youden's J). The default 0.5
                      boundary called 73.6% of the official test "pneumonia"
                      (prevalence 62.5%) because the ranking is good (AUC
                      0.975) but 0.5 sits far below the model's operating point;
  - tau_low/tau_high: trust levels on the confidence in the predicted label.

Usage: python t3_agentic/eval/fit_thresholds.py
"""
import json
import os
import sys

import numpy as np

ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

from sklearn.metrics import (accuracy_score, average_precision_score,
                             confusion_matrix, roc_auc_score)
from sklearn.model_selection import train_test_split

from t3_agentic.agent.calibration import apply_temperature
from t3_agentic.eval.calibration import ece

SEED = 42
PREDS = os.path.join(ROOT, "outputs", "preds_kermany_full.npz")
OUT = os.path.join(ROOT, "outputs")

# criteria for the fitted operating point (documented in the report)
TRUST_ACC = 0.95     # TAU_LOW : below this, the image is not trusted (refuse)
STRONG_ACC = 0.99    # TAU_HIGH: above this, confidence is called "haute"
STRONG_COVERAGE = 0.30
THRESHOLD_GRID = np.arange(0.05, 0.995, 0.005)


def confidence(p, y, threshold):
    """Confidence in the PREDICTED label."""
    return np.where(p >= threshold, p, 1.0 - p)


def fit_temperature(p, y):
    """Minimise NLL on the validation half (logit scaling)."""
    from scipy.optimize import minimize_scalar
    eps = 1e-7
    lg = np.log(np.clip(p, eps, 1 - eps) / (1 - np.clip(p, eps, 1 - eps)))

    def nll(log_t):
        t = float(np.exp(log_t))
        z = lg / t
        return float(-np.mean(y * np.log(1 / (1 + np.exp(-z))) +
                              (1 - y) * np.log(1 / (1 + np.exp(z)))))

    res = minimize_scalar(nll, bounds=(-2.0, 2.0), method="bounded")
    return float(np.exp(res.x))


def balanced_stats(p, y, t):
    """Accuracy / balanced accuracy / recalls / predicted positive rate at t."""
    pred = (p >= t).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    n = len(y)
    tpr = tp / (tp + fn) if tp + fn else 0.0
    tnr = tn / (tn + fp) if tn + fp else 0.0
    return {
        "threshold": round(float(t), 3),
        "accuracy": round(float((pred == y).mean()), 4),
        "balanced_accuracy": round(float(0.5 * (tpr + tnr)), 4),
        "recall_pneumonia": round(float(tpr), 4),
        "recall_normal": round(float(tnr), 4),
        "precision_pneumonia": round(float(tp / (tp + fp)), 4) if tp + fp else None,
        "predicted_positive_rate": round(float(pred.mean()), 4),
        "false_positive": int(fp), "false_negative": int(fn),
    }


def fit_label_threshold(p_cal, y, grid=THRESHOLD_GRID):
    """Threshold on the CALIBRATED score maximising Youden's J (TPR - FPR).

    Ties are broken towards the HIGHER threshold (more specific), because the
    default 0.5 boundary over-called pneumonia and the false positives are
    what a clinician notices.
    """
    best_t, best_j = 0.5, -1.0
    for t in grid:
        pred = (p_cal >= t).astype(int)
        tp = float(((pred == 1) & (y == 1)).sum())
        fn = float(((pred == 0) & (y == 1)).sum())
        tn = float(((pred == 0) & (y == 0)).sum())
        fp = float(((pred == 1) & (y == 0)).sum())
        j = (tp / (tp + fn) if tp + fn else 0.0) - (fp / (fp + tn) if fp + tn else 0.0)
        if j > best_j + 1e-12 or (abs(j - best_j) <= 1e-12 and t > best_t):
            best_t, best_j = float(t), j
    return best_t


def fit_tau_low(conf, pred, y, min_acc=TRUST_ACC):
    """LEAST restrictive threshold t such that accuracy of the TRUSTED set is >= min_acc.

    conf: confidence in the predicted label (ranking key)
    pred: the model's decision (1 = pneumonia) - NOT derived from conf
    """
    order = np.argsort(-conf)
    cs, ys, ps = conf[order], y[order], pred[order]
    correct = (ps == ys).astype(float)
    cum_acc = np.cumsum(correct) / np.arange(1, len(ys) + 1)
    bad = np.where(cum_acc < min_acc)[0]
    k = int(bad[0]) - 1 if len(bad) else len(cs) - 1
    return float(cs[max(k, 0)])


def fit_tau_high(conf, pred, y, min_acc=STRONG_ACC, min_cov=STRONG_COVERAGE):
    """LEAST restrictive threshold t with accuracy >= min_acc AND coverage >= min_cov."""
    order = np.argsort(-conf)
    cs, ys, ps = conf[order], y[order], pred[order]
    correct = (ps == ys).astype(float)
    cum_acc = np.cumsum(correct) / np.arange(1, len(ys) + 1)
    cov = np.arange(1, len(ys) + 1) / len(ys)
    k = len(cs) - 1
    for i in range(len(cs)):
        if cum_acc[i] < min_acc or cov[i] < min_cov:
            k = i - 1
            break
    return float(cs[max(k, 0)])


def raw_equivalent(p_cal_threshold, temperature):
    """The raw P(pneumonia) that maps to a calibrated threshold."""
    t = min(max(float(p_cal_threshold), 1e-7), 1 - 1e-7)
    lg = np.log(t / (1 - t)) * temperature
    return float(1 / (1 + np.exp(-lg)))


def main():
    d = np.load(PREDS)
    p, y, names = d["p"], d["y"], d["names"]
    idx = np.arange(len(p))
    val_idx, test_idx = train_test_split(idx, test_size=0.5, stratify=y,
                                         random_state=SEED)
    pv, yv = p[val_idx], y[val_idx]
    pt, yt = p[test_idx], y[test_idx]

    # ---- fit on VALIDATION ----
    temperature = fit_temperature(pv, yv)
    pv_cal = np.array([apply_temperature(x, temperature) for x in pv])
    label_threshold = fit_label_threshold(pv_cal, yv)
    conf_v = confidence(pv_cal, yv, label_threshold)
    pred_v = (pv_cal >= label_threshold).astype(int)
    tau_low = fit_tau_low(conf_v, pred_v, yv)
    tau_high = fit_tau_high(conf_v, pred_v, yv)

    # ---- report on TEST ----
    pt_cal = np.array([apply_temperature(x, temperature) for x in pt])
    conf_t = confidence(pt_cal, yt, label_threshold)
    pred = (pt_cal >= label_threshold).astype(int)
    trusted = conf_t >= tau_low
    ece_before, _ = ece(pt, yt)
    ece_after, _ = ece(pt_cal, yt)

    # trade-off curve (validation half): the evidence behind the threshold
    sweep = []
    for t in np.arange(0.50, 1.001, 0.05):
        s = balanced_stats(pv_cal, yv, t)
        s["step"] = "val"
        sweep.append(s)
    sweep_test = [dict(balanced_stats(pt_cal, yt, t), step="test")
                  for t in np.arange(0.50, 1.001, 0.05)]

    report = {
        "protocol": {
            "source": "test officiel Kermany (624 images, jamais vu a l'entrainement)",
            "split": "stratified 312/312, seed 42",
            "fitted_on": "validation half (312) uniquement",
            "reported_on": "test half (312)",
            "n_val": int(len(pv)), "n_test": int(len(pt)),
            "criteria": {"tau_low_accuracy": TRUST_ACC,
                         "tau_high_accuracy": STRONG_ACC,
                         "tau_high_min_coverage": STRONG_COVERAGE,
                         "label_threshold": "max balanced accuracy (Youden J), "
                                            "ties -> higher threshold"},
            "nih": "jamais utilise pour l'ajustement (controle de decalage non ajuste)",
        },
        "fitted": {
            "label_threshold": round(label_threshold, 4),
            "label_threshold_equivalent_raw_p": round(
                raw_equivalent(label_threshold, temperature), 4),
            "tau_low": round(tau_low, 4),
            "tau_high": round(tau_high, 4),
            "temperature": round(temperature, 4),
        },
        "operating_point": {
            "validation_half": balanced_stats(pv_cal, yv, label_threshold),
            "test_half": balanced_stats(pt_cal, yt, label_threshold),
            "official_test_full_624": balanced_stats(
                np.array([apply_temperature(x, temperature) for x in p]), y,
                label_threshold),
            "official_test_full_624_at_default_0.5": balanced_stats(
                np.array([apply_temperature(x, temperature) for x in p]), y, 0.5),
        },
        "validation_half": {
            "accuracy_raw": round(accuracy_score(yv, (pv >= 0.5).astype(int)), 4),
            "accuracy_calibrated_at_fitted": round(
                accuracy_score(yv, (pv_cal >= label_threshold).astype(int)), 4),
            "auc": round(roc_auc_score(yv, pv), 4),
        },
        "test_half": {
            "accuracy_raw_at_0.5": round(accuracy_score(yt, (pt >= 0.5).astype(int)), 4),
            "accuracy_calibrated_at_fitted": round(
                accuracy_score(yt, (pt_cal >= label_threshold).astype(int)), 4),
            "auc": round(roc_auc_score(yt, pt), 4),
            "pr_auc": round(average_precision_score(yt, pt), 4),
            "ece_before_temperature": round(ece_before, 4),
            "ece_after_temperature": round(ece_after, 4),
            "coverage_trusted": round(float(trusted.mean()), 4),
            "accuracy_on_trusted": round(
                accuracy_score(yt[trusted], pred[trusted]) if trusted.any() else None, 4),
            "abstained": int((~trusted).sum()),
            "confusion_matrix": confusion_matrix(yt, pred).tolist(),
            "confusion_matrix_trusted_only": confusion_matrix(
                yt[trusted], pred[trusted]).tolist() if trusted.any() else None,
        },
        "threshold_sweep_validation": sweep,
        "threshold_sweep_test": sweep_test,
        "caveat": "312 images per half: wide confidence intervals; the thresholds "
                  "are indicative and must be re-fitted on a larger set.",
    }
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "threshold_fit.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    with open(os.path.join(OUT, "calibration_config.json"), "w", encoding="utf-8") as f:
        json.dump({"label_threshold": round(label_threshold, 4),
                   "tau_low": round(tau_low, 4), "tau_high": round(tau_high, 4),
                   "temperature": round(temperature, 4),
                   "source": "fitted on the Kermany official-test validation half "
                             "(312 images, seed 42)",
                   "fitted_on": "Kermany val half", "never_fitted_on": "NIH"},
                  f, ensure_ascii=False, indent=2)
    np.savez(os.path.join(OUT, "preds_split.npz"), p=p, y=y, names=names,
             val_idx=val_idx, test_idx=test_idx)
    print(json.dumps({k: report[k] for k in ("fitted", "operating_point")},
                     ensure_ascii=False, indent=2))
    return report


if __name__ == "__main__":
    main()
