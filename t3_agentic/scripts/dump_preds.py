"""Dump per-image probabilities for calibration/ECE (inference only, no retraining).

Usage:
  python t3_agentic/scripts/dump_preds.py --dir demo_images/clean_kermany/test --out outputs/preds_kermany.npz
"""
import argparse
import os
import sys

import numpy as np

ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

from t3_agentic.vision.model import classify


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--preprocessed", action="store_true",
                    help="les images ont deja passe par preprocess_gray (PNG du notebook)")
    ap.add_argument("--max-per-class", type=int, default=0,
                    help="limite le nombre d'images par classe (0 = toutes)")
    args = ap.parse_args()

    files = []
    for dirpath, _, names in os.walk(args.dir):
        for n in sorted(names):
            if n.lower().endswith((".png", ".jpg", ".jpeg")):
                files.append(os.path.join(dirpath, n))
    files.sort()
    if args.max_per_class:
        kept, counts = [], {}
        for f in files:
            cls = os.path.basename(os.path.dirname(f)).upper()
            if counts.get(cls, 0) >= args.max_per_class:
                continue
            counts[cls] = counts.get(cls, 0) + 1
            kept.append(f)
        files = kept
        print("selection:", counts)
    ps, ys, names = [], [], []
    for i, f in enumerate(files):
        try:
            label, p, _ = classify(f, already_preprocessed=args.preprocessed)
        except Exception as e:
            print(f"skip {f}: {e}")
            continue
        # ground truth comes from the FOLDER name, never from the prediction
        cls = os.path.basename(os.path.dirname(f)).upper()
        gt = 1 if cls == "PNEUMONIA" else 0
        ps.append(p)
        ys.append(gt)
        names.append(os.path.basename(f))
        if (i + 1) % 50 == 0:
            print(f"{i+1}/{len(files)}", flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.savez(args.out, p=np.array(ps), y=np.array(ys), names=np.array(names))
    print("saved", args.out, len(ps), "images")


if __name__ == "__main__":
    main()
