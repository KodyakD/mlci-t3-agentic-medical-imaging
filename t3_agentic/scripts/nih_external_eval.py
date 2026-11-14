"""Phase 2 : per-image NIH predictions with the Kermany-fitted calibration.

The Kaggle kernel did not save per-image NIH probabilities and the kernel-output
API is rate limited, so we rerun the inference LOCALLY on the public NIH
Chest X-ray14 *sample* (same source as the kernel): no GPU, no Kaggle quota.

Corpus: demo_images/nih_eval/<label>/*.png  (downloaded once from the dataset;
the per-file Kaggle CLI download is unreliable, so the corpus is the source of
truth and the label comes from the folder name).

Label definition (identical to the trained model):
  positive = Pneumonia | Consolidation | Infiltration
  negative = No Finding
NOTE: the NIH sample contains only 14 strictly-"Pneumonia" images; positives
are dominated by Infiltration, which is NOT pneumonia. Reported, not hidden.

Outputs: outputs/preds_nih.npz, outputs/nih_metrics.json
"""
import json
import os
import sys

import numpy as np

ROOT = r"C:\Users\PC\Desktop\M2S1\MLCI"
sys.path.insert(0, ROOT)

CORPUS = os.path.join(ROOT, "demo_images", "nih_eval")
OUT = os.path.join(ROOT, "outputs")
POSITIVE_DIRS = {"Pneumonia", "Consolidation", "Infiltration"}


def corpus():
    items = []
    if not os.path.isdir(CORPUS):
        return items
    for cls in sorted(os.listdir(CORPUS)):
        d = os.path.join(CORPUS, cls)
        if not os.path.isdir(d):
            continue
        for f in sorted(os.listdir(d)):
            if f.lower().endswith(".png"):
                items.append((os.path.join(d, f), cls))
    return items


def main():
    items = corpus()
    if not items:
        print("ABORT: empty NIH corpus (run the download step first)", flush=True)
        return
    n_pos = sum(1 for _, c in items if c in POSITIVE_DIRS)
    print(f"corpus: {len(items)} images ({n_pos} positive / {len(items)-n_pos} negative)", flush=True)
    if n_pos < 20 or (len(items) - n_pos) < 20:
        print("ABORT: not enough images per class", flush=True)
        return

    from t3_agentic.vision.model import get_model, preprocess_file, to_input
    model = get_model()
    grays, ys, names, lbls = [], [], [], []
    for i, (path, cls) in enumerate(items):
        try:
            grays.append(preprocess_file(path))
        except Exception:
            continue
        ys.append(1 if cls in POSITIVE_DIRS else 0)
        names.append(os.path.basename(path))
        lbls.append(cls)
        if (i + 1) % 100 == 0:
            print(f"preprocessed {i+1}/{len(items)}", flush=True)

    probs, batch = [], []
    for g in grays:
        batch.append(to_input(g))
        if len(batch) == 32:
            probs.extend(model.predict(np.stack(batch), verbose=0).ravel().tolist())
            batch = []
    if batch:
        probs.extend(model.predict(np.stack(batch), verbose=0).ravel().tolist())
    p = np.array(probs)
    y = np.array(ys)

    np.savez(os.path.join(OUT, "preds_nih.npz"), p=p, y=y,
             names=np.array(names), labels=np.array(lbls))

    from sklearn.metrics import (accuracy_score, average_precision_score,
                                 balanced_accuracy_score, roc_auc_score)
    from t3_agentic.agent.calibration import apply_temperature, load_config
    from t3_agentic.eval.calibration import ece
    cfg = load_config()
    p_cal = np.array([apply_temperature(x, cfg["temperature"]) for x in p])
    pred = (p_cal >= 0.5).astype(int)
    conf = np.where(p_cal >= 0.5, p_cal, 1 - p_cal)
    trusted = conf >= cfg["tau_low"]
    e, _ = ece(p_cal, y)
    metrics = {
        "n": int(len(p)),
        "class_balance": {"negatives": int((y == 0).sum()), "positives": int((y == 1).sum())},
        "label_definition": "positive = Pneumonia|Consolidation|Infiltration ; negative = No Finding",
        "label_breakdown": {c: int(sum(1 for l in lbls if l == c)) for c in sorted(set(lbls))},
        "calibration": {"tau_low": cfg["tau_low"], "tau_high": cfg["tau_high"],
                        "temperature": cfg["temperature"],
                        "fitted_on": "Kermany validation half only (unfitted on NIH)"},
        "accuracy": round(accuracy_score(y, pred), 4),
        "balanced_accuracy": round(balanced_accuracy_score(y, pred), 4),
        "auc": round(roc_auc_score(y, p_cal), 4),
        "pr_auc": round(average_precision_score(y, p_cal), 4),
        "ece": round(e, 4),
        "recall_normal": round(float(((pred == 0) & (y == 0)).sum() / max((y == 0).sum(), 1)), 4),
        "recall_pneumonia_like": round(float(((pred == 1) & (y == 1)).sum() / max((y == 1).sum(), 1)), 4),
        "trusted_coverage": round(float(trusted.mean()), 4),
        "trusted_accuracy": round(float((pred[trusted] == y[trusted]).mean()), 4) if trusted.any() else None,
        "trusted_n": int(trusted.sum()),
        "majority_baseline_accuracy": round(float(max((y == 0).mean(), (y == 1).mean())), 4),
    }
    cand = [(names[i], lbls[i], float(p[i]), float(p_cal[i]))
            for i in range(len(p)) if y[i] == 1 and pred[i] == 0]
    cand.sort(key=lambda t: -t[3])
    metrics["confident_normal_on_positive_top10"] = [
        {"image": c[0], "label": c[1], "p_raw": round(c[2], 4), "p_cal": round(c[3], 4)}
        for c in cand[:10]]
    with open(os.path.join(OUT, "nih_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(json.dumps({k: v for k, v in metrics.items()
                      if k != "confident_normal_on_positive_top10"},
                     ensure_ascii=False, indent=2))
    print("\ntop confident-normal mistakes on positives:")
    for c in cand[:5]:
        print(f"  {c[0]} ({c[1]}) p_raw={c[2]:.3f} p_cal={c[3]:.3f}")


if __name__ == "__main__":
    main()
