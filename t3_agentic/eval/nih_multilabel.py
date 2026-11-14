"""Multi-label analysis of the NIH domain-shift check.

The deployed CNN is BINARY (pneumonia-like opacity vs normal). The NIH labels
are MULTI-LABEL (14 findings mined from free-text reports). The binary model
therefore has to be judged in two ways, both reported here:

1. LABEL COLLAPSE (what the merged training used):
   positive = Pneumonia | Consolidation | Infiltration, negative = No Finding.
   -> shows the collapse explicitly and how noisy the positives are.

2. PER-LABEL RANKING (what the binary score can still do):
   for every finding label present in the sample, the CNN probability is used
   as a *ranking* signal for that finding: AUC + precision/recall at the
   Kermany-fitted trust threshold. A binary model can rank a finding it was
   never trained on, even if it cannot threshold it.

Per-image label strings come from sample_labels.csv (the source of truth); the
folder name in demo_images/nih_eval/ is only used to locate the image.

Outputs: outputs/nih_multilabel.json
"""
import json
import os
import sys

import numpy as np

ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

# Two corpora:
#  - nih_eval       : the binary-collapse corpus (single-label images + No Finding)
#  - nih_multilabel : stratified multi-label sample (see build_nih_multilabel.py)
CORPUS = os.path.join(ROOT, "demo_images", "nih_eval")
CORPUS_ML = os.path.join(ROOT, "demo_images", "nih_multilabel")
MANIFEST = os.path.join(CORPUS_ML, "labels_manifest.json")
CSV = os.path.join(ROOT, "demo_images", "raw", "nih", "sample_labels.csv")
OUT = os.path.join(ROOT, "outputs")
POSITIVE_FINDINGS = ("Pneumonia", "Consolidation", "Infiltration")
ALL_FINDINGS = ["Atelectasis", "Cardiomegaly", "Effusion", "Infiltration",
                "Mass", "Nodule", "Pneumonia", "Pneumothorax",
                "Consolidation", "Edema", "Emphysema", "Fibrosis",
                "Pleural_Thickening", "Hernia"]
NO_FINDING = "No Finding"


def label_map():
    """Image Index -> list of findings."""
    import csv as csvmod
    out = {}
    with open(CSV, encoding="utf-8") as f:
        for row in csvmod.DictReader(f):
            parts = [p.strip() for p in str(row["Finding Labels"]).split("|")]
            out[row["Image Index"]] = parts
    return out


def corpus_images():
    """All NIH images available locally, deduplicated by file name.

    Returns {filename: absolute path}. The multi-label manifest wins over the
    CSV (it carries the exact label string used at download time).
    """
    files = {}
    for root_dir in (CORPUS, CORPUS_ML):
        if not os.path.isdir(root_dir):
            continue
        for cls in sorted(os.listdir(root_dir)):
            d = os.path.join(root_dir, cls)
            if not os.path.isdir(d):
                continue
            for f in sorted(os.listdir(d)):
                if f.lower().endswith(".png"):
                    files.setdefault(f, os.path.join(d, f))
    return files


def main():
    from sklearn.metrics import roc_auc_score, average_precision_score, precision_score, recall_score
    from t3_agentic.vision.model import get_model, preprocess_file, to_input
    from t3_agentic.agent.calibration import load_config

    lm = label_map()
    manifest = {}
    if os.path.exists(MANIFEST):
        manifest = json.load(open(MANIFEST, encoding="utf-8"))
    files = corpus_images()
    items = sorted(files.items())
    print(f"corpus: {len(items)} images "
          f"({len(manifest)} with a multi-label manifest)", flush=True)
    if len(items) < 50:
        print("ABORT: NIH corpus too small")
        return

    model = get_model()
    cfg = load_config()
    tau_low = float(cfg.get("TAU_LOW", 0.75))

    probs, kept_files = [], []
    for i, (name, path) in enumerate(items):
        try:
            g = preprocess_file(path)
        except Exception:
            continue
        kept_files.append((name, path))
        probs.append(float(model.predict(np.expand_dims(to_input(g), 0), verbose=0)[0][0]))
        if (i + 1) % 100 == 0:
            print(f"predicted {i+1}/{len(items)}", flush=True)
    p = np.array(probs)

    findings_per_image = []
    for name, _ in kept_files:
        if name in manifest:
            parts = [p.strip() for p in manifest[name].split("|")]
        else:
            parts = lm.get(name, [])
        findings_per_image.append(parts)
    n = len(p)
    print(f"labelled images: {n}", flush=True)
    n_nofinding = sum(1 for x in findings_per_image if x == [NO_FINDING])
    print(f"  No Finding: {n_nofinding} | with findings: {n - n_nofinding}", flush=True)

    # ---------------- 1. label collapse -------------------------------------
    collapsed = []
    for parts in findings_per_image:
        if parts == [NO_FINDING]:
            collapsed.append(0)
        elif any(x in POSITIVE_FINDINGS for x in parts):
            collapsed.append(1)
        else:
            collapsed.append(-1)   # neither: dropped by the collapse
    collapsed = np.array(collapsed)
    kept = collapsed >= 0
    y_col = collapsed[kept]
    p_col = p[kept]
    pred_col = (p_col >= 0.5).astype(int)
    coll = {
        "n_images": int(n),
        "n_kept_by_collapse": int(kept.sum()),
        "n_dropped_other_findings": int((collapsed == -1).sum()),
        "collapse_accuracy": round(float((pred_col == y_col).mean()), 4),
        "collapse_auc": round(float(roc_auc_score(y_col, p_col)), 4),
        "recall_pneumonia_like": round(float(recall_score(y_col, pred_col)), 4),
        "recall_normal": round(float(recall_score(1 - y_col, 1 - pred_col)), 4),
    }
    # how noisy are the "positives"? (other findings co-occurring)
    pos_parts = [parts for parts, c in zip(findings_per_image, collapsed) if c == 1]
    with_other = sum(1 for parts in pos_parts
                      if any(x not in POSITIVE_FINDINGS for x in parts))
    coll["positives_with_at_least_one_other_finding"] = int(with_other)
    coll["positives_total"] = int(len(pos_parts))
    coll["positives_with_other_finding_ratio"] = round(
        with_other / max(len(pos_parts), 1), 4)
    coll["strict_pneumonia_only"] = int(
        sum(1 for parts in pos_parts if set(parts) == {"Pneumonia"}))

    # ---------------- 2. per-label ranking ---------------------------------
    per_label = []
    for lab in ALL_FINDINGS:
        y = np.array([1 if lab in parts else 0 for parts in findings_per_image])
        if y.sum() < 5 or (1 - y).sum() < 5:
            continue
        try:
            auc = float(roc_auc_score(y, p))
            ap = float(average_precision_score(y, p))
        except ValueError:
            continue
        pred = (p >= 0.5).astype(int)
        per_label.append({
            "finding": lab,
            "n_positive": int(y.sum()),
            "prevalence": round(float(y.mean()), 4),
            "auc_ranking": round(auc, 4),
            "average_precision": round(ap, 4),
            "precision_at_0.5": round(float(precision_score(y, pred, zero_division=0)), 4),
            "recall_at_0.5": round(float(recall_score(y, pred, zero_division=0)), 4),
            "recall_at_tau_low": round(
                float(recall_score(y, (p >= tau_low).astype(int), zero_division=0)), 4),
            "trust_threshold_used": tau_low,
        })
    per_label.sort(key=lambda d: -d["auc_ranking"])
    macro_auc = round(float(np.mean([d["auc_ranking"] for d in per_label])), 4) if per_label else None

    # ---------------- 3. co-occurrence --------------------------------------
    multi = sum(1 for parts in findings_per_image if len(parts) > 1 and parts != [NO_FINDING])
    cooc = {
        "images_with_several_findings": int(multi),
        "ratio_multi_label": round(multi / max(n, 1), 4),
        "mean_findings_per_image": round(
            float(np.mean([len(parts) for parts in findings_per_image])), 4),
    }

    report = {"n_images": int(n),
              "tau_low_kermany_fitted": tau_low,
              "label_collapse": coll,
              "per_label_ranking": per_label,
              "macro_auc_over_findings": macro_auc,
              "cooccurrence": cooc}
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "nih_multilabel.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(json.dumps({"label_collapse": coll, "macro_auc_over_findings": macro_auc,
                      "cooccurrence": cooc}, ensure_ascii=False, indent=2))
    print("\nper-label ranking (sorted by AUC):")
    for d in per_label:
        print(f"  {d['finding']:22s} n+={d['n_positive']:4d}  AUC={d['auc_ranking']:.3f}  "
              f"AP={d['average_precision']:.3f}  R@0.5={d['recall_at_0.5']:.3f}  "
              f"R@tau={d['recall_at_tau_low']:.3f}")


if __name__ == "__main__":
    main()
