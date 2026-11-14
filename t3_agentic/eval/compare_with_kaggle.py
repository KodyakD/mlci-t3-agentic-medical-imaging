"""Compare the LOCAL pipeline against the Kaggle reference numbers.

The merged model was trained in a Kaggle notebook whose preprocessing is
reproduced here (crop 5 %, resize 256 INTER_AREA, percentile 1-99 clip, CLAHE).
If the local run reproduces the Kaggle metrics on the official Kermany test
split, the local pipeline is faithful; a mismatch means a preprocessing drift.
"""
import json
import os
import sys

import numpy as np

ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

# Reference: Kaggle kernel abdelraoufgheyat/kermany-nih-merged, Kermany test (940 imgs)
KAGGLE = {"accuracy": 0.9660, "auc": 0.9893}
# Same kernel, official Kermany test split (624 imgs) is a subset -> expected close but not identical
TOLERANCE = {"accuracy": 0.03, "auc": 0.02}


def auc(y, p):
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(np.asarray(y).ravel(), np.asarray(p).ravel()))


def main(path="outputs/preds_kermany.npz"):
    data = np.load(os.path.join(ROOT, path))
    p, y = data["p"], data["y"]
    acc = float(((p >= 0.5).astype(int) == y).mean())
    a = auc(y, p)
    report = {
        "n": int(len(p)),
        "local_accuracy": round(acc, 4),
        "local_auc": round(a, 4),
        "kaggle_accuracy": KAGGLE["accuracy"],
        "kaggle_auc": KAGGLE["auc"],
        "delta_accuracy": round(acc - KAGGLE["accuracy"], 4),
        "delta_auc": round(a - KAGGLE["auc"], 4),
        "within_tolerance": bool(abs(acc - KAGGLE["accuracy"]) <= TOLERANCE["accuracy"]
                                 and abs(a - KAGGLE["auc"]) <= TOLERANCE["auc"]),
        "split_note": "local = test officiel Kermany (624 imgs, JAMAIS vu a l'entrainement) ; "
                      "Kaggle = re-split patient du notebook (940 imgs, issu du pool train+val). "
                      "Les deux chiffres sont vrais mais ne se comparent pas directement.",
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))
    out = os.path.join(ROOT, "outputs", "local_vs_kaggle.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    return report


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "outputs/preds_kermany.npz")
