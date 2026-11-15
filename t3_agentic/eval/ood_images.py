"""Out-of-distribution sanity check: run the vision tools on any folder of images.

The evaluation sets (Kermany official test, NIH) are the measured ones. This
script exists for the OTHER kind of evidence: images that come from nowhere
near the training distribution (adult web radiographs, annotated teaching
files, portable AP films, watermarked images). It reports what the deployed
operating point does on them, including the Grad-CAM plausibility check.

Usage:
  python t3_agentic/eval/ood_images.py --dir path/to/images --out outputs/ood_images.json
"""
import argparse
import glob
import json
import os
import sys
import time

ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

from t3_agentic.agent.mcp_client import MCPPool


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="folder of radiographs")
    ap.add_argument("--out", default=os.path.join(ROOT, "outputs", "ood_images.json"))
    args = ap.parse_args()

    files = sorted(f for f in glob.glob(os.path.join(args.dir, "*"))
                   if f.lower().endswith((".png", ".jpg", ".jpeg")))
    if not files:
        print("no image found in", args.dir)
        return

    pool = MCPPool()
    rows = []
    for f in files:
        t0 = time.time()
        cls = pool.call_tool("classify_image", {"path": f}).get("result", {})
        exp = pool.call_tool("explain_image", {"path": f}).get("result", {})
        row = {
            "file": os.path.basename(f),
            "label": cls.get("label"),
            "p_pneumonia_raw": cls.get("p_pneumonia"),
            "p_pneumonia_calibrated": cls.get("p_pneumonia_calibrated"),
            "label_threshold": cls.get("label_threshold"),
            "confidence_in_label": cls.get("score"),
            "inside_ratio": exp.get("inside_ratio"),
            "gradcam_suspect": exp.get("suspect"),
            "seconds": round(time.time() - t0, 2),
        }
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)

    report = {
        "protocol": {
            "images": len(rows),
            "source": args.dir,
            "note": "out-of-distribution sanity check, NOT a benchmark: no ground "
                    "truth, images may carry watermarks, annotations, portable "
                    "technique or letterboxing that the training preprocessing "
                    "(Kermany paediatric JPEGs, NIH adults) never saw",
            "operating_point": "Kermany-fitted (never fitted on this corpus)",
        },
        "results": rows,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2)
    print("written:", args.out)


if __name__ == "__main__":
    main()
