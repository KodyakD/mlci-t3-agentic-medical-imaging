"""Calibration: ECE + reliability curve from saved probabilities (no retraining).

Usage:
  python t3_agentic/eval/calibration.py --npz outputs/preds_kermany.npz outputs/preds_nih.npz
"""
import argparse
import json
import os

import numpy as np

ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUT = os.path.join(ROOT, "outputs")


def ece(p, y, bins=10):
    """Expected calibration error for P(pneumonia) and binary labels."""
    p = np.asarray(p).ravel()
    y = np.asarray(y).ravel().astype(int)
    conf = np.where(p >= 0.5, p, 1 - p)
    pred = (p >= 0.5).astype(int)
    acc = (pred == y).astype(float)
    edges = np.linspace(0.5, 1.0, bins + 1)
    total, rows = 0.0, []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if not m.any():
            continue
        bin_acc = float(acc[m].mean())
        bin_conf = float(conf[m].mean())
        total += (m.sum() / len(conf)) * abs(bin_acc - bin_conf)
        rows.append({"bin": [round(float(lo), 3), round(float(hi), 3)],
                     "n": int(m.sum()), "acc": round(bin_acc, 4),
                     "conf": round(bin_conf, 4)})
    return float(total), rows


def reliability_plot(curves, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 5))
    for name, rows in curves.items():
        if not rows:
            continue
        ax.plot([r["conf"] for r in rows], [r["acc"] for r in rows], marker="o", label=name)
    ax.plot([0.5, 1.0], [0.5, 1.0], "k--", label="perfect calibration")
    ax.set_xlabel("predicted confidence")
    ax.set_ylabel("observed frequency")
    ax.set_title("Reliability curve")
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npz", nargs="+", help="fichiers .npz avec cles p et y")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    curves, report = {}, {}
    for path in args.npz:
        name = os.path.splitext(os.path.basename(path))[0]
        data = np.load(path)
        p, y = data["p"], data["y"]
        e, rows = ece(p, y)
        curves[name] = rows
        report[name] = {"n": int(len(p)), "ece": round(e, 4),
                         "accuracy": round(float(((p >= .5).astype(int) == y).mean()), 4),
                         "bins": rows}
        print(f"{name}: n={len(p)} ECE={e:.4f} acc={report[name]['accuracy']}")
    with open(os.path.join(OUT, "calibration.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    plot = reliability_plot(curves, os.path.join(OUT, "reliability_curve.png"))
    print("plot:", plot)


if __name__ == "__main__":
    main()
