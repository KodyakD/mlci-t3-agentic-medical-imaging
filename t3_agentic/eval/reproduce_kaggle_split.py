"""Rebuild the Kaggle notebook's Kermany split locally, and verify it.

Why this exists: the merged training notebook (kermany_nih_merged.ipynb, cell 1)
collects Kermany's train, val AND test folders and re-splits the union by
patient, so the OFFICIAL test folder was part of the training pool of
densenet_merged.keras. The notebook's own test split is patient-disjoint from
its training data and is therefore the only clean held-out slice of that model.

This script reproduces the notebook's split (same file order, same md5 dedup on
the preprocessed bytes, same GroupShuffleSplit with seed 42), then runs the
deployed weights on the reproduced test slice. If the accuracy/AUC match the
numbers the Kaggle run reported (0.9660 / 0.9893), the reproduction is proven and
the slice is a legitimate held-out set.

Usage:
  python t3_agentic/eval/reproduce_kaggle_split.py --root <path to chest_xray>
"""
import argparse
import hashlib
import json
import os
import random
import re
import sys
import time

import cv2
import numpy as np
from sklearn.model_selection import GroupShuffleSplit

ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

from t3_agentic.vision.model import get_onnx, preprocess_file

SEED = 42
CLASSES = ["NORMAL", "PNEUMONIA"]


def kermany_patient_id(cls, fname):
    """Same rule as the notebook (cell 1)."""
    m = re.match(r"person(\d+)_", fname) if cls == "PNEUMONIA" else re.search(r"IM-(\d+)", fname)
    return cls + (m.group(1) if m else fname)


def collect(root):
    """Same order as the notebook: split order, then class order, sorted names."""
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
                    pids.append(kermany_patient_id(c, f))
    return files, np.array(labels), np.array(pids)


def dedup(files, labels, pids):
    """md5 of the PREPROCESSED bytes, first occurrence wins (notebook build_split)."""
    seen, keep = {}, []
    t0 = time.time()
    for i, (f, lab, pid) in enumerate(zip(files, labels, pids)):
        a = preprocess_file(f)
        if a is None:
            continue
        h = hashlib.md5(a.tobytes()).hexdigest()
        if h in seen:
            continue
        seen[h] = i
        keep.append(i)
        if (i + 1) % 500 == 0:
            print(f"  preprocessed {i+1}/{len(files)}", flush=True)
    print(f"  dedup: {len(files)} -> {len(keep)} in {time.time()-t0:.0f}s", flush=True)
    return np.array(keep)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="chest_xray folder (train/val/test)")
    ap.add_argument("--out", default=os.path.join(ROOT, "outputs", "kaggle_split_reproduction.json"))
    ap.add_argument("--expect-acc", type=float, default=0.9660)
    ap.add_argument("--expect-auc", type=float, default=0.9893)
    args = ap.parse_args()

    files, y_all, pids = collect(args.root)
    print(f"collected {len(files)} files", flush=True)
    keep = dedup(files, y_all, pids)
    y = y_all[keep]
    groups = pids[keep]

    # notebook: GroupShuffleSplit(0.15) -> train/test ; then 0.1765 of train -> val
    gs = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=SEED)
    tr_idx, te_idx = next(gs.split(np.zeros(len(keep)), y, groups=groups))
    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.1765, random_state=SEED)
    tr_idx2, va_idx2 = next(gss2.split(np.zeros(len(tr_idx)), y[tr_idx], groups=groups[tr_idx]))
    splits = {"train": tr_idx[tr_idx2], "val": tr_idx[va_idx2], "test": te_idx}

    report = {"pool_after_dedup": int(len(keep)),
              "splits": {k: int(len(v)) for k, v in splits.items()},
              "class_counts": {k: {c: int((y[v] == i).sum()) for i, c in enumerate(CLASSES)}
                               for k, v in splits.items()}}

    # patient-level disjointness (the whole point of the split)
    sets = {k: set(groups[v]) for k, v in splits.items()}
    report["patient_overlap"] = {
        "train_val": len(sets["train"] & sets["val"]),
        "train_test": len(sets["train"] & sets["test"]),
        "val_test": len(sets["val"] & sets["test"]),
    }
    print(json.dumps(report, indent=1), flush=True)

    # ---- run the deployed weights on the reproduced VAL and TEST slices ----
    sess = get_onnx()
    if sess is None:
        print("no ONNX session: cannot score the slice")
        return

    def score(idx):
        out = []
        for i in idx:
            x = np.expand_dims(
                np.repeat(preprocess_file(files[keep[i]]).astype("float32")[..., None] / 255.0,
                          3, axis=-1), 0).astype("float32")
            out.append(float(sess.run(None, {"input": x})[0].flatten()[0]))
        return np.array(out)

    t0 = time.time()
    p_val = score(splits["val"])
    p_test = score(splits["test"])
    y_val, y_test = y[splits["val"]], y[splits["test"]]
    print(f"  scored {len(p_val)} val + {len(p_test)} test in {time.time()-t0:.0f}s",
          flush=True)

    # saved for the threshold fit: val = fitting set, test = reporting set,
    # both patient-disjoint from the model's training data.
    np.savez(os.path.join(ROOT, "outputs", "preds_kermany_clean_split.npz"),
             p=np.concatenate([p_val, p_test]),
             y=np.concatenate([y_val, y_test]),
             names=np.array([os.path.basename(files[keep[i]])
                             for i in list(splits["val"]) + list(splits["test"])]),
             split=np.array(["val"] * len(p_val) + ["test"] * len(p_test)),
             paths=np.array([files[keep[i]]
                             for i in list(splits["val"]) + list(splits["test"])]),
             patient=np.array([groups[i]
                               for i in list(splits["val"]) + list(splits["test"])]))
    with open(os.path.join(ROOT, "outputs", "kaggle_test_files.json"), "w",
              encoding="utf-8") as f:
        json.dump([files[keep[i]] for i in splits["test"]], f, indent=1)

    from sklearn.metrics import accuracy_score, roc_auc_score
    acc = float(accuracy_score(y_test, (p_test > 0.5).astype(int)))
    auc = float(roc_auc_score(y_test, p_test))
    report["scored_test_slice"] = {
        "n": int(len(y_test)),
        "val_slice_n": int(len(y_val)),
        "val_slice_accuracy_at_0.5": round(
            float(accuracy_score(y_val, (p_val > 0.5).astype(int))), 4),
        "val_slice_auc": round(float(roc_auc_score(y_val, p_val)), 4),
        "accuracy_at_0.5": round(acc, 4),
        "auc": round(auc, 4),
        "kaggle_run_logged_accuracy": args.expect_acc,
        "kaggle_run_logged_auc": args.expect_auc,
        "matches_kaggle_run": bool(abs(acc - args.expect_acc) < 0.01
                                   and abs(auc - args.expect_auc) < 0.01),
        "scoring_seconds": round(time.time() - t0, 1),
    }
    print(json.dumps(report["scored_test_slice"], indent=1), flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print("written:", args.out)


if __name__ == "__main__":
    main()
