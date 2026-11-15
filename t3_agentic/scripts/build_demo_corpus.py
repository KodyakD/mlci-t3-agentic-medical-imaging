"""Build the demo/test image corpus from the public Kaggle datasets.

Portable: works on Windows / Linux / macOS, no hardcoded paths.

Why raw images: the merged model's preprocessing is deterministic, so
preprocess_gray(raw_jpeg) == the preprocessed PNG the training notebook wrote.

READ THIS BEFORE CALLING THESE IMAGES "HELD OUT": the merged training notebook
(kermany_nih_merged.ipynb, cell 1) collects Kermany's train, val AND test
folders and re-splits the union by patient, so for densenet_merged.keras the
official test folder was part of the training pool (verified: the Kaggle
kernel source is identical to the notebook in this repo). This corpus is a
REPRODUCTION corpus -- same preprocessing, same images -- not a clean hold-out.
The clean, patient-disjoint held-out slice of that model is the notebook's own
test split, rebuilt and verified by eval/reproduce_kaggle_split.py.

Strategy (avoids the paginated `datasets files` listing):
  - Kermany: one zip download of the whole dataset, keep chest_xray/test/ only.
  - NIH   : download sample_labels.csv, pick image names by label, fetch them
            one by one into a unique temp folder (avoids a race when several
            downloads share a destination directory).

Requirements: the `kaggle` package and a free Kaggle account
(`pip install kaggle` then put your kaggle.json in ~/.kaggle/).
This step is OPTIONAL: the evaluation numbers are already in outputs/.
"""
import csv as csvmod
import os
import shutil
import subprocess
import sys
import time
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, "..", ".."))
DEMO = os.path.join(ROOT, "demo_images")
RAW = os.path.join(DEMO, "raw")
KAGGLE = shutil.which("kaggle") or "kaggle"

POSITIVE_DIRS = {"Pneumonia", "Consolidation", "Infiltration"}


def kaggle(*args):
    r = subprocess.run([KAGGLE, *args], capture_output=True, text=True, timeout=3600)
    if r.returncode != 0:
        raise RuntimeError((r.stderr or r.stdout)[-500:])
    return r.stdout


def download_one(dataset, filename, dest_dir, dest_name, tries=3):
    """Download one file, each into its own temp folder to avoid races."""
    os.makedirs(dest_dir, exist_ok=True)
    target = os.path.join(dest_dir, dest_name)
    if os.path.exists(target) and os.path.getsize(target) > 0:
        return target
    tmp = os.path.join(RAW, "_dl")
    os.makedirs(tmp, exist_ok=True)
    for attempt in range(tries):
        kaggle("datasets", "download", dataset, "-f", filename, "-p", tmp, "-q")
        found = None
        for root, _, files in os.walk(tmp):
            if os.path.basename(filename) in files:
                found = os.path.join(root, os.path.basename(filename))
                break
        if found:
            shutil.move(found, target)
            return target
        time.sleep(3 * (attempt + 1))
    return None


def kermany(n_test=None):
    """One zip download, keep the official test/ split only."""
    test_root = os.path.join(DEMO, "kermany_test")
    if os.path.isdir(test_root) and sum(len(f) for _, _, f in os.walk(test_root)) > 100:
        print("kermany: already present")
        return
    os.makedirs(RAW, exist_ok=True)
    print("kermany: downloading the dataset zip (~1.2 GB) ...", flush=True)
    kaggle("datasets", "download", "paultimothymooney/chest-xray-pneumonia", "-p", RAW, "-q")
    zips = [os.path.join(RAW, f) for f in os.listdir(RAW) if f.endswith(".zip")]
    if not zips:
        raise RuntimeError("Kermany zip not found after download")
    n = 0
    with zipfile.ZipFile(zips[0]) as z:
        for member in z.namelist():
            parts = member.replace("\\", "/").split("/")
            if len(parts) < 4 or parts[0] != "chest_xray" or parts[1] != "test":
                continue
            cls, fname = parts[2], parts[3]
            if cls not in ("NORMAL", "PNEUMONIA") or fname.startswith("._"):
                continue
            dest = os.path.join(test_root, cls)
            os.makedirs(dest, exist_ok=True)
            with z.open(member) as src, open(os.path.join(dest, fname), "wb") as dst:
                shutil.copyfileobj(src, dst)
            n += 1
    print(f"kermany: {n} test images extracted")


def nih(n_per_class=4):
    """Optional: a handful of NIH adult images (domain-shift check)."""
    csv_path = os.path.join(RAW, "nih", "sample_labels.csv")
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    if not os.path.exists(csv_path):
        kaggle("datasets", "download", "nih-chest-xrays/sample",
               "-f", "sample/sample_labels.csv", "-p", os.path.dirname(csv_path), "-q")
    rows = list(csvmod.DictReader(open(csv_path, encoding="utf-8")))
    pos = [r["Image Index"] for r in rows if r["Finding Labels"].strip() in POSITIVE_DIRS]
    neg = [r["Image Index"] for r in rows if r["Finding Labels"].strip() == "No Finding"]
    for cls, names in (("PNEUMONIA", pos[:n_per_class]), ("NORMAL", neg[:n_per_class])):
        dest = os.path.join(DEMO, "nih_test", cls)
        for i, n in enumerate(names):
            download_one("nih-chest-xrays/sample", f"sample/images/{n}", dest, f"{cls}_{i}.png")
    print("nih: sample written to demo_images/nih_test")


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "kermany"
    if which in ("kermany", "all"):
        kermany()
    if which in ("nih", "all"):
        nih()
    print("corpus in", DEMO)
