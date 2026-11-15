"""Phase 5 / B3 : robustness of the vision component to image perturbations.

CPU-only, no retraining: we perturb the held-out Kermany test images and measure
how the predictions and the decision move.

Perturbations: gaussian noise (sigma=10/255), rotation (+/-10 deg), contrast (x0.6).

Outputs: outputs/robustness.json
"""
import json
import os
import sys

import cv2
import numpy as np

ROOT = r"C:\Users\PC\Desktop\M2S1\MLCI"
sys.path.insert(0, ROOT)

from sklearn.metrics import roc_auc_score

from t3_agentic.agent.calibration import apply_temperature, load_config
from t3_agentic.eval.calibration import ece
from t3_agentic.vision.model import get_model, preprocess_file, to_input

IMG_DIR = os.path.join(ROOT, "demo_images", "kermany_test")
OUT = os.path.join(ROOT, "outputs")
SEED = 42


def perturb_gray(a, kind, rng):
    if kind == "noise":
        noise = rng.normal(0, 10.0 / 255.0, a.shape).astype(np.float32)
        return np.clip(a.astype(np.float32) / 255.0 + noise, 0, 1) * 255.0
    if kind == "rotation":
        h, w = a.shape
        M = cv2.getRotationMatrix2D((w / 2, h / 2), 10.0, 1.0)
        return cv2.warpAffine(a, M, (w, h), flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_REFLECT_101)
    if kind == "contrast":
        x = a.astype(np.float32) / 255.0
        return np.clip((x - 0.5) * 0.6 + 0.5, 0, 1) * 255.0
    return a.astype(np.float32)


def main():
    cfg = load_config()
    model = get_model()
    rng = np.random.default_rng(SEED)

    files, ys = [], []
    for cls in ("NORMAL", "PNEUMONIA"):
        d = os.path.join(IMG_DIR, cls)
        for f in sorted(os.listdir(d)):
            files.append(os.path.join(d, f))
            ys.append(1 if cls == "PNEUMONIA" else 0)
    y = np.array(ys)
    print(f"{len(files)} images de test", flush=True)

    def predict(grays):
        xs = np.stack([to_input(g) for g in grays])
        return model.predict(xs, verbose=0).ravel()

    clean = []
    for i, f in enumerate(files):
        clean.append(preprocess_file(f))
        if (i + 1) % 100 == 0:
            print(f"preprocessed {i+1}/{len(files)}", flush=True)
    p_clean = predict(clean)

    def stats(p, name, flip=None):
        p_cal = np.array([apply_temperature(x, cfg["temperature"]) for x in p])
        t = cfg["label_threshold"]
        pred = (p_cal >= t).astype(int)
        conf = np.where(p_cal >= t, p_cal, 1 - p_cal)
        trusted = conf >= cfg["tau_low"]
        e, _ = ece(p_cal, y)
        out = {
            "perturbation": name,
            "accuracy": round(float((pred == y).mean()), 4),
            "auc": round(float(roc_auc_score(y, p_cal)), 4),
            "ece": round(float(e), 4),
            "trusted_coverage": round(float(trusted.mean()), 4),
            "trusted_accuracy": round(float((pred[trusted] == y[trusted]).mean()), 4) if trusted.any() else None,
        }
        if flip is not None:
            out["decision_flip_rate"] = round(float(flip), 4)
        return out, p_cal

    results = []
    c_stat, p_clean_cal = stats(p_clean, "aucune (reference)")
    results.append(c_stat)
    for kind in ("noise", "rotation", "contrast"):
        perturbed = [perturb_gray(a, kind, rng) for a in clean]
        p = predict(perturbed)
        p_cal = np.array([apply_temperature(x, cfg["temperature"]) for x in p])
        flip = float((((p_cal >= cfg["label_threshold"]).astype(int))
                      != ((p_clean_cal >= cfg["label_threshold"]).astype(int))).mean())
        s, _ = stats(p, kind, flip)
        results.append(s)
        print(json.dumps(s, ensure_ascii=False), flush=True)

    report = {
        "protocol": {
            "n": len(files),
            "source": "test officiel Kermany (624 images), non ajuste",
            "perturbations": {"noise": "gaussien sigma=10/255",
                              "rotation": "+10 degres (bord reflect)",
                              "contrast": "x0.6 autour de 0.5"},
            "calibration": {"tau_low": cfg["tau_low"], "tau_high": cfg["tau_high"],
                            "temperature": cfg["temperature"],
                            "label_threshold": cfg["label_threshold"]},
        },
        "results": results,
    }
    with open(os.path.join(OUT, "robustness.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
