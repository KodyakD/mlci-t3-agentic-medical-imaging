"""Did the deployed model memorise its training set?

The merged notebook's pool contains Kermany's official test folder, so the
question is whether those images are memorised (near-perfect accuracy, extreme
scores) or merely trained on (no signature). This script rebuilds the
notebook's split, scores the TRAIN slice with the deployed weights, and
compares it with the test slice.

Usage:
  python t3_agentic/eval/memorisation_check.py --root <path to chest_xray>
"""
import argparse
import hashlib
import json
import os
import re
import sys
import time

import numpy as np
from sklearn.model_selection import GroupShuffleSplit

ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

from t3_agentic.vision.model import get_onnx, preprocess_file

CLASSES = ["NORMAL", "PNEUMONIA"]
SEED = 42
MAP = os.path.join(ROOT, "outputs", "kaggle_split_map.json")


def patient_id(cls, fname):
    m = re.match(r"person(\d+)_", fname) if cls == "PNEUMONIA" else re.search(r"IM-(\d+)", fname)
    return cls + (m.group(1) if m else fname)


def build_map(root):
    """file -> notebook split (patient-level), same order as the notebook."""
    files, labels, pids = [], [], []
    for split in ("train", "val", "test"):
        for ci, c in enumerate(CLASSES):
            d = os.path.join(root, split, c)
            if not os.path.isdir(d):
                continue
            for f in sorted(os.listdir(d)):
                if f.lower().endswith((".jpeg", ".jpg", ".png")):
                    files.append(os.path.join(d, f))
                    labels.append(ci)
                    pids.append(patient_id(c, f))
    first = {}
    for i, f in enumerate(files):
        a = preprocess_file(f)
        h = hashlib.md5(a.tobytes()).hexdigest()
        first.setdefault(h, i)
    keep = np.array(sorted(first.values()))
    y = np.array(labels)[keep]
    groups = np.array(pids)[keep]
    gs = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=SEED)
    tr, te = next(gs.split(np.zeros(len(keep)), y, groups=groups))
    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.1765, random_state=SEED)
    tr2, va2 = next(gss2.split(np.zeros(len(tr)), y[tr], groups=groups[tr]))
    out = {}
    for i in tr[tr2]:
        out[files[keep[i]]] = ["train", int(y[np.where(keep == keep[i])[0][0]])]
    for i in tr[va2]:
        out[files[keep[i]]] = ["val", int(y[np.where(keep == keep[i])[0][0]])]
    for i in te:
        out[files[keep[i]]] = ["test", int(y[np.where(keep == keep[i])[0][0]])]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    args = ap.parse_args()

    if os.path.exists(MAP):
        with open(MAP, encoding="utf-8") as f:
            smap = json.load(f)
        print(f"loaded split map ({len(smap)} files)")
    else:
        t0 = time.time()
        smap = build_map(args.root)
        with open(MAP, "w", encoding="utf-8") as f:
            json.dump(smap, f)
        print(f"built split map ({len(smap)} files) in {time.time()-t0:.0f}s")

    sess = get_onnx()
    if sess is None:
        print("no ONNX session")
        return

    def score(path):
        x = np.expand_dims(
            np.repeat(preprocess_file(path).astype("float32")[..., None] / 255.0,
                      3, axis=-1), 0).astype("float32")
        return float(sess.run(None, {"input": x})[0].flatten()[0])

    out = {}
    for target in ("train", "test"):
        files = [f for f, (s, _) in smap.items() if s == target]
        t0 = time.time()
        ps = np.array([score(f) for f in files])
        ys = np.array([smap[f][1] for f in files])
        acc = float(((ps > 0.5).astype(int) == ys).mean())
        out[target] = {"n": len(files), "accuracy_at_0.5": round(acc, 4),
                       "mean_abs_distance_from_0.5": round(float(np.abs(ps - 0.5).mean()), 4),
                       "median_p": round(float(np.median(ps)), 4),
                       "seconds": round(time.time() - t0, 1)}
        print(f"{target:6s} n={len(files):5d} accuracy@0.5 = {acc:.4f} "
              f"|p-0.5| = {np.abs(ps-0.5).mean():.4f} median p = {np.median(ps):.4f} "
              f"({time.time()-t0:.0f}s)", flush=True)

    gap = out["train"]["accuracy_at_0.5"] - out["test"]["accuracy_at_0.5"]
    out["train_minus_test_accuracy"] = round(gap, 4)
    out["memorisation"] = bool(gap > 0.05)
    out["reading"] = ("the model does not memorise its training slice: the "
                      "accuracy on images it trained on is not materially "
                      "higher than on the unseen slice, so the official test "
                      "folder being inside the pool does not translate into "
                      "inflated scores on those images" if not out["memorisation"]
                      else "the model memorises its training slice")
    print(json.dumps(out, indent=1))
    with open(os.path.join(ROOT, "outputs", "memorisation_check.json"), "w",
              encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
