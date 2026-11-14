"""Download a stratified MULTI-LABEL NIH corpus (for the per-label analysis).

The eval corpus (demo_images/nih_eval) only contains images whose label string
is a single finding, because the collapse training only needed those. The CSV
has 980 multi-label rows (e.g. "Effusion|Infiltration") plus the rarer findings
(Cardiomegaly, Atelectasis, Emphysema...) that the binary model never saw.

This script downloads a stratified sample:
  - for every finding: up to --per-label images labelled with it
  - the multi-label rows are preferred, so co-occurrence is represented
Images keep their original NIH filename (e.g. 00000116_019.png) and the full
label string is stored in a sidecar JSON (labels_manifest.json).

Usage: python t3_agentic/scripts/build_nih_multilabel.py --per-label 25
Requires the kaggle CLI and a free Kaggle account.
"""
import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import time

ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))
RAW = os.path.join(ROOT, "demo_images", "raw", "nih")
DEST = os.path.join(ROOT, "demo_images", "nih_multilabel")
CSV_PATH = os.path.join(RAW, "sample_labels.csv")
KAGGLE = shutil.which("kaggle") or "kaggle"

ALL_FINDINGS = ["Atelectasis", "Cardiomegaly", "Effusion", "Infiltration",
                "Mass", "Nodule", "Pneumonia", "Pneumothorax",
                "Consolidation", "Edema", "Emphysema", "Fibrosis",
                "Pleural_Thickening", "Hernia"]


def kaggle(*args):
    r = subprocess.run([KAGGLE, *args], capture_output=True, text=True, timeout=3600)
    if r.returncode != 0:
        raise RuntimeError((r.stderr or r.stdout)[-400:])
    return r.stdout


def download_one(filename, dest_dir, tries=3):
    os.makedirs(dest_dir, exist_ok=True)
    target = os.path.join(dest_dir, filename)
    if os.path.exists(target) and os.path.getsize(target) > 0:
        return target
    tmp = os.path.join(RAW, "_dl")
    os.makedirs(tmp, exist_ok=True)
    for attempt in range(tries):
        try:
            kaggle("datasets", "download", "nih-chest-xrays/sample",
                   "-f", f"sample/images/{filename}", "-p", tmp, "-q")
        except RuntimeError:
            pass
        for root, _, files in os.walk(tmp):
            if filename in files:
                shutil.move(os.path.join(root, filename), target)
                return target
        time.sleep(2 * (attempt + 1))
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-label", type=int, default=25)
    args = ap.parse_args()

    if not os.path.exists(CSV_PATH):
        print("ABORT: sample_labels.csv missing", flush=True)
        return
    rows = list(csv.DictReader(open(CSV_PATH, encoding="utf-8")))
    os.makedirs(DEST, exist_ok=True)

    # group by finding, preferring multi-label rows
    by_finding = {f: [] for f in ALL_FINDINGS}
    for r in rows:
        parts = [p.strip() for p in r["Finding Labels"].split("|")]
        if parts == ["No Finding"]:
            continue
        for p in parts:
            if p in by_finding:
                by_finding[p].append((len(parts), r["Image Index"], r["Finding Labels"]))

    manifest = {}
    n_ok = n_fail = 0
    for finding, items in by_finding.items():
        # multi-label first (they carry the co-occurrence information)
        items.sort(key=lambda t: -t[0])
        picked = 0
        for nparts, name, labstr in items:
            if picked >= args.per_label:
                break
            d = os.path.join(DEST, finding.replace(" ", "_"))
            if download_one(name, d) is None:
                n_fail += 1
                continue
            manifest[name] = labstr
            picked += 1
            n_ok += 1
        print(f"{finding:22s} downloaded {picked}", flush=True)

    with open(os.path.join(DEST, "labels_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(f"total {n_ok} images, {n_fail} failures -> {DEST}")


if __name__ == "__main__":
    main()
