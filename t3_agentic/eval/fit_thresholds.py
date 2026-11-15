"""Fit the label threshold, the trust threshold and the temperature.

PROTOCOL (disclosed in the report):
  - source: the Kermany dataset as the Kaggle training notebook used it;
  - the notebook re-splits Kermany train+val+test BY PATIENT, so the official
    test folder was inside the training pool of densenet_merged.keras. The
    clean, patient-disjoint slice is rebuilt locally by
    eval/reproduce_kaggle_split.py (verified: the rebuilt test slice reproduces
    the Kaggle run log exactly, accuracy 0.9660 / AUC 0.9893);
  - the temperature, the decision threshold and the trust threshold are fitted
    on the clean VALIDATION slice only;
  - everything is reported on the clean TEST slice;
  - NIH is never used for fitting (it stays the unfitted domain-shift check).

What is fitted:
  - temperature      : minimises NLL (logit scaling) on the validation slice;
  - label_threshold  : the decision boundary on the CALIBRATED score, chosen to
                       maximise balanced accuracy (Youden's J). At the default
                       0.5 the model called 73.6% of the official test set
                       "pneumonia" (prevalence 62.5%) because the ranking is
                       good (AUC 0.975) but 0.5 sits far below the model's
                       operating point;
  - TAU_LOW          : the MIRROR of the decision boundary -- a label is
                       asserted only if the calibrated confidence in it reaches
                       the operating point's own requirement. The least
                       restrictive threshold that keeps 95% accuracy on the
                       trusted set would be ~0.28, but that rule trusts calls
                       the model itself is only ~50% sure of, which turns coin
                       flips on out-of-domain images into "normal" (false
                       reassurance). Measured, not assumed;
  - TAU_HIGH         : confidence above which the answer is called "haute"
                       (>=99% accuracy and >=30% coverage on the validation set).

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
OUT = os.path.join(ROOT, "outputs")
CLEAN = os.path.join(OUT, "preds_kermany_clean_split.npz")
LEGACY = os.path.join(OUT, "preds_kermany_full.npz")

# criteria for the fitted operating point (documented in the report)
TRUST_ACC = 0.95     # reference criterion for the permissive trust variant
STRONG_ACC = 0.99    # TAU_HIGH: above this confidence is called "haute"
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
    """LEAST restrictive threshold t such that accuracy of the TRUSTED set is >= min_acc."""
    order = np.argsort(-conf)
    cs, ys, ps = conf[order], y[order], pred[order]
    correct = (ps == ys).astype(float)
    cum_acc = np.cumsum(correct) / np.arange(1, len(ys) + 1)
    bad = np.where(cum_acc < min_acc)[0]
    k = int(bad[0]) - 1 if len(bad) else len(cs) - 1
    return float(cs[max(k, 0)])


def fit_tau_high(conf, pred, y, min_acc=STRONG_ACC, min_cov=STRONG_COVERAGE):
    """LEAST restrictive threshold t with accuracy >= min_acc AND coverage >= min_cov.

    (The previous version stopped at the first sample of the sorted list, which
    always has a coverage far below min_cov, so it returned the MAXIMUM
    confidence -- the "haute confiance" path was unreachable in practice.)
    """
    order = np.argsort(-conf)
    cs, ys, ps = conf[order], y[order], pred[order]
    correct = (ps == ys).astype(float)
    cum_acc = np.cumsum(correct) / np.arange(1, len(ys) + 1)
    cov = np.arange(1, len(ys) + 1) / len(ys)
    ok = np.where((cum_acc >= min_acc) & (cov >= min_cov))[0]
    if not len(ok):
        return float(cs[0])
    return float(cs[ok[-1]])


def raw_equivalent(p_cal_threshold, temperature):
    """The raw P(pneumonia) that maps to a calibrated threshold."""
    t = min(max(float(p_cal_threshold), 1e-7), 1 - 1e-7)
    lg = np.log(t / (1 - t)) * temperature
    return float(1 / (1 + np.exp(-lg)))


def main():
    if os.path.exists(CLEAN):
        d = np.load(CLEAN, allow_pickle=True)
        p, y, names = d["p"], d["y"], d["names"]
        split = d["split"]
        val_idx = np.where(split == "val")[0]
        test_idx = np.where(split == "test")[0]
        protocol = ("clean patient-disjoint split rebuilt from the Kaggle run "
                    "(eval/reproduce_kaggle_split.py): fitted on val (871), "
                    "reported on test (940)")
        source = "outputs/preds_kermany_clean_split.npz"
    else:  # legacy: stratified 312/312 split of the official test predictions
        d = np.load(LEGACY)
        p, y, names = d["p"], d["y"], d["names"]
        idx = np.arange(len(p))
        val_idx, test_idx = train_test_split(idx, test_size=0.5, stratify=y,
                                             random_state=SEED)
        protocol = ("stratified 312/312 split (seed 42) of the official test "
                    "predictions")
        source = "outputs/preds_kermany_full.npz"

    pv, yv = p[val_idx], y[val_idx]
    pt, yt = p[test_idx], y[test_idx]

    # ---- fit on the VALIDATION slice ----
    temperature = fit_temperature(pv, yv)
    pv_cal = np.array([apply_temperature(x, temperature) for x in pv])
    label_threshold = fit_label_threshold(pv_cal, yv)
    conf_v = confidence(pv_cal, yv, label_threshold)
    pred_v = (pv_cal >= label_threshold).astype(int)
    tau_low_permissive = fit_tau_low(conf_v, pred_v, yv)
    tau_low = label_threshold          # the mirror rule (see the module docstring)
    tau_high = fit_tau_high(conf_v, pred_v, yv)

    # ---- report on the TEST slice ----
    pt_cal = np.array([apply_temperature(x, temperature) for x in pt])
    conf_t = confidence(pt_cal, yt, label_threshold)
    pred = (pt_cal >= label_threshold).astype(int)
    trusted = conf_t >= tau_low
    ece_before, _ = ece(pt, yt, threshold=0.5)
    ece_after, _ = ece(pt_cal, yt)

    # the selective behaviour: assert only outside the indeterminate zone
    assert_pneu = pt_cal >= label_threshold
    assert_norm = pt_cal <= (1 - label_threshold)
    answered = assert_pneu | assert_norm

    sweep = [dict(balanced_stats(pv_cal, yv, t), step="val")
             for t in np.arange(0.50, 1.001, 0.05)]
    sweep_test = [dict(balanced_stats(pt_cal, yt, t), step="test")
                  for t in np.arange(0.50, 1.001, 0.05)]

    report = {
        "protocol": {
            "source": source,
            "split": protocol,
            "fitted_on": f"validation slice ({len(pv)} images), patient-disjoint",
            "reported_on": f"test slice ({len(pt)} images), patient-disjoint",
            "n_val": int(len(pv)), "n_test": int(len(pt)),
            "criteria": {"label_threshold": "max balanced accuracy (Youden J), "
                                             "ties -> higher threshold",
                         "tau_low": "mirror of the decision boundary",
                         "tau_high_accuracy": STRONG_ACC,
                         "tau_high_min_coverage": STRONG_COVERAGE},
            "nih": "jamais utilise pour l'ajustement (controle de decalage non ajuste)",
        },
        "fitted": {
            "label_threshold": round(label_threshold, 4),
            "label_threshold_equivalent_raw_p": round(
                raw_equivalent(label_threshold, temperature), 4),
            "tau_low": round(tau_low, 4),
            "tau_low_permissive_variant": round(tau_low_permissive, 4),
            "tau_high": round(tau_high, 4),
            "temperature": round(temperature, 4),
        },
        "operating_point": {
            "validation_slice": balanced_stats(pv_cal, yv, label_threshold),
            "test_slice": balanced_stats(pt_cal, yt, label_threshold),
            "test_slice_at_default_0.5": balanced_stats(pt_cal, yt, 0.5),
        },
        "selective_behaviour": {
            "asserted_pneumonia": int(assert_pneu.sum()),
            "asserted_normal": int(assert_norm.sum()),
            "abstained_indeterminate": int((~answered).sum()),
            "coverage_answered": round(float(answered.mean()), 4),
            "accuracy_on_answered": round(
                float((pred[answered] == yt[answered]).mean()), 4),
            "accuracy_pneumonia_asserted": round(
                float((pred[assert_pneu] == yt[assert_pneu]).mean()), 4),
            "accuracy_normal_asserted": round(
                float((pred[assert_norm] == yt[assert_norm]).mean()), 4),
        },
        "validation_slice": {
            "accuracy_raw": round(accuracy_score(yv, (pv >= 0.5).astype(int)), 4),
            "accuracy_calibrated_at_fitted": round(
                accuracy_score(yv, (pv_cal >= label_threshold).astype(int)), 4),
            "auc": round(roc_auc_score(yv, pv), 4),
        },
        "test_slice": {
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
        },
        "threshold_sweep_validation": sweep,
        "threshold_sweep_test": sweep_test,
        "caveat": "the validation slice is 871 images: the intervals are "
                  "narrower than the earlier 312-image split, but the operating "
                  "point is still fitted on one dataset and should be re-fitted "
                  "on more data before any clinical use.",
    }
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "threshold_fit.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    with open(os.path.join(OUT, "calibration_config.json"), "w", encoding="utf-8") as f:
        json.dump({"label_threshold": round(label_threshold, 4),
                   "tau_low_rule": "mirror of label_threshold",
                   "tau_low": round(tau_low, 4), "tau_high": round(tau_high, 4),
                   "temperature": round(temperature, 4),
                   "source": "fitted on the clean Kermany validation slice "
                             "(patient-disjoint, rebuilt from the Kaggle split)",
                   "fitted_on": "Kermany clean validation slice",
                   "never_fitted_on": "NIH"},
                  f, ensure_ascii=False, indent=2)
    print(json.dumps({k: report[k] for k in ("protocol", "fitted", "operating_point",
                                             "selective_behaviour")},
                     ensure_ascii=False, indent=2))
    return report


if __name__ == "__main__":
    main()
