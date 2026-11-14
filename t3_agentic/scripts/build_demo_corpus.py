"""Build the demo/test image corpus from public Kaggle DATASETS (not kernel output).

Why raw images: the merged model's preprocessing is deterministic, so
preprocess_gray(raw_jpeg) == the preprocessed PNG the training kernel wrote.
The official Kermany test/ folder was never used for training (the merged
notebook split only the train+val pool), so these are genuinely held-out.

Strategy (avoids the paginated `datasets files` listing):
  - Kermany: one zip download of the whole dataset, then keep chest_xray/test/ only.
  - NIH    : download sample_labels.csv, pick image names by label, fetch those files.
"""
import csv as csvmod
import os
import re
import shutil
import subprocess
import zipfile

KAGGLE = r"C:\Users\PC\AppData\Roaming\Python\Python313\Scripts\kaggle.EXE"
ROOT = r"C:\Users\PC\Desktop\M2S1\MLCI"
DEMO = os.path.join(ROOT, "demo_images")
RAW = os.path.join(DEMO, "raw")
KERMANY_ZIP = os.path.join(RAW, "kermany.zip")


def kaggle(*args):
    r = subprocess.run([KAGGLE, *args], capture_output=True, text=True, timeout=3600)
    if r.returncode != 0:
        raise RuntimeError((r.stderr or r.stdout)[-400:])
    return r.stdout


def download_file(dataset, filename, dest_dir, dest_name):
    os.makedirs(dest_dir, exist_ok=True)
    target = os.path.join(dest_dir, dest_name)
    if os.path.exists(target) and os.path.getsize(target) > 0:
        return target
    kaggle("datasets", "download", dataset, "-f", filename, "-p", dest_dir, "-q")
    if not os.path.exists(target):
        base = os.path.basename(filename)
        for root, _, files in os.walk(dest_dir):
            if base in files:
                os.replace(os.path.join(root, base), target)
                break
    return target if os.path.exists(target) else None


def kermany():
    """Download the dataset zip once, keep the official test/ split."""
    test_root = os.path.join(DEMO, "kermany_test")
    if os.path.isdir(test_root) and sum(len(f) for _, _, f in os.walk(test_root)) > 100:
        print("kermany: deja present", flush=True)
        return
    if not os.path.exists(KERMANY_ZIP):
        print("kermany: telechargement du zip (~1.2 Go)...", flush=True)
        kaggle("datasets", "download", "paultimothymooney/chest-xray-pneumonia",
               "-p", RAW, "-q")
    zip_path = KERMANY_ZIP
    if not os.path.exists(zip_path):
        cands = [f for f in os.listdir(RAW) if f.endswith(".zip")]
        if not cands:
            raise RuntimeError("zip Kermany introuvable")
        zip_path = os.path.join(RAW, cands[0])
    n = 0
    with zipfile.ZipFile(zip_path) as z:
        for member in z.namelist():
            m = re.match(r"chest_xray/test/(NORMAL|PNEUMONIA)/([^\s]+\.jpe?g)$", member)
            if not m or m.group(2).startswith("._"):
                continue
            cls, fname = m.group(1), m.group(2)
            dest = os.path.join(test_root, cls)
            os.makedirs(dest, exist_ok=True)
            with z.open(member) as src, open(os.path.join(dest, fname), "wb") as dst:
                shutil.copyfileobj(src, dst)
            n += 1
    print("kermany: images test extraites =", n, flush=True)


def nih(n_per_class=4):
    dest_root = os.path.join(DEMO, "nih_test")
    csv_path = os.path.join(RAW, "nih", "sample_labels.csv")
    if not os.path.exists(csv_path):
        download_file("nih-chest-xrays/sample", "sample/sample_labels.csv",
                      os.path.join(RAW, "nih"), "sample_labels.csv")
    rows = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csvmod.DictReader(f):
            rows.append(row)
    pneu = [r["Image Index"] for r in rows if r.get("Finding Labels") == "Pneumonia"][:n_per_class]
    norm = [r["Image Index"] for r in rows if r.get("Finding Labels") == "No Finding"][:n_per_class]
    for cls, picks in (("PNEUMONIA", pneu), ("NORMAL", norm)):
        dest = os.path.join(dest_root, cls)
        os.makedirs(dest, exist_ok=True)
        for i, n in enumerate(picks):
            download_file("nih-chest-xrays/sample", f"sample/images/{n}",
                          os.path.join(RAW, "nih"), f"{cls}_{i}.png")
        print("nih", cls, len(picks), flush=True)


if __name__ == "__main__":
    os.makedirs(RAW, exist_ok=True)
    try:
        kermany()
    except Exception as e:
        print("kermany failed:", e, flush=True)
    try:
        nih()
    except Exception as e:
        print("nih failed:", e, flush=True)
    print("demo images:", sum(len(f) for _, _, f in os.walk(DEMO)), flush=True)
