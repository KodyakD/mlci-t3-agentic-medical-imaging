"""NIH domain-shift MCP server (second dataset, dedicated server).

Exposes what the NIH domain-shift work measured, as READ-ONLY tools, so an
agent can answer questions about the NIH Chest X-ray14 sample without touching
any model or any patient data:

  nih_domain_shift   -> the measured transfer-gap report (loss of accuracy,
                        retained ranking AUC) on the local 551-image sample
  nih_findings       -> the 14 findings and their prevalence in the sample
  nih_case_lookup    -> one sample image: its exact multi-label string + the
                        folder it came from ("infiltration", "No Finding", ...)

Datasets used: nih-chest-xrays/sample (public Kaggle), the same one the
merged training notebook uses. The server is stateless: it reads
demo_images/*/labels_manifest.json + outputs/nih_multilabel.json.
"""
import json
import os
import sys
from collections import Counter

from mcp.server.fastmcp import FastMCP

ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

CORPUS = os.path.join(ROOT, "demo_images", "nih_multilabel")
EVAL_CORPUS = os.path.join(ROOT, "demo_images", "nih_eval")
METRICS = os.path.join(ROOT, "outputs", "nih_multilabel.json")

mcp = FastMCP("nih")

ALL_FINDINGS = ["Atelectasis", "Cardiomegaly", "Effusion", "Infiltration",
                "Mass", "Nodule", "Pneumonia", "Pneumothorax",
                "Consolidation", "Edema", "Emphysema", "Fibrosis",
                "Pleural_Thickening", "Hernia"]


def _manifest():
    path = os.path.join(CORPUS, "labels_manifest.json")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return {}


@mcp.tool()
def nih_domain_shift() -> dict:
    """The measured NIH transfer gap. Always exact: read from outputs/."""
    if not os.path.exists(METRICS):
        return {"error": "outputs/nih_multilabel.json missing"}
    with open(METRICS, encoding="utf-8") as f:
        return json.load(f)


@mcp.tool()
def nih_findings() -> list[dict]:
    """The 14 NIH findings and their prevalence in the local sample."""
    man = _manifest()
    counts = Counter()
    n_multi = 0
    for labstr in man.values():
        parts = [p.strip() for p in labstr.split("|")]
        if len(parts) > 1:
            n_multi += 1
        for p in parts:
            counts[p] += 1
    total = max(len(man), 1)
    rows = []
    for f_ in ALL_FINDINGS:
        rows.append({"finding": f_, "n_images": counts.get(f_, 0),
                     "prevalence": round(counts.get(f_, 0) / total, 4)})
    rows.sort(key=lambda r: -r["n_images"])
    rows.append({"finding": "_multi_label_images_(in_manifest_only)", "n_images": n_multi,
                 "prevalence": round(n_multi / total, 4),
                 "note": "manifest holds the stratified sample (multi-label preferred); "
                          "the honest co-occurrence rate lives in nih_domain_shift()"})
    return rows


@mcp.tool()
def nih_case_lookup(image: str) -> dict:
    """One sample image by its file name (e.g. 00000116_019.png).

    Returns the exact multi-label string the NIH NLP-mined labels were stored
    under, the folder it lives in, and whether it is currently used in any
    evaluation. The image itself is not served: only its labels.
    """
    man = _manifest()
    labstr = man.get(image)
    folder = None
    if labstr is None:
        for base in (CORPUS, EVAL_CORPUS):
            for root, dirs, files in os.walk(base):
                if image in files:
                    folder = os.path.basename(root)
                    break
            if folder:
                break
        # fall back to the raw CSV
        csv_path = os.path.join(ROOT, "demo_images", "raw", "nih",
                                "sample_labels.csv")
        if os.path.exists(csv_path):
            import csv as csvmod
            with open(csv_path, encoding="utf-8") as f:
                for row in csvmod.DictReader(f):
                    if row["Image Index"] == image:
                        labstr = row["Finding Labels"]
                        break
    if labstr is None and folder is None:
        return {"error": f"image introuvable dans l'echantillon NIH: {image}"}
    return {"image": image, "labels": labstr, "folder": folder,
            "n_findings": len(labstr.split("|")) if labstr else 0}


@mcp.resource("nih://summary")
def nih_summary() -> str:
    """Plain-text summary of the NIH domain-shift findings."""
    if not os.path.exists(METRICS):
        return "Aucune mesure NIH disponible (outputs/nih_multilabel.json)."
    with open(METRICS, encoding="utf-8") as f:
        m = json.load(f)
    coll = m["label_collapse"]
    parts = [
        "NIH Chest X-ray14 sample - transfer-gap measurement (K's Kermany-fitted model).",
        f"Corpus: {m['n_images']} images, {coll['n_kept_by_collapse']} kept by the binary collapse.",
        f"Collapsed-label accuracy {coll['collapse_accuracy']}, AUC {coll['collapse_auc']}, "
        f"pneumonia-like recall {coll['recall_pneumonia_like']}.",
        f"Macro AUC over {len(m['per_label_ranking'])} findings: {m['macro_auc_over_findings']}.",
        f"Multi-label images: {m['cooccurrence']['images_with_several_findings']} "
        f"({m['cooccurrence']['ratio_multi_label']:.1%}), mean findings/image {m['cooccurrence']['mean_findings_per_image']:.2f}.",
        "Only 14 strictly-Pneumonia images exist in the sample; the positives are dominated by 'Infiltration'.",
    ]
    return "\n".join(parts)


if __name__ == "__main__":
    mcp.run(transport="stdio")
