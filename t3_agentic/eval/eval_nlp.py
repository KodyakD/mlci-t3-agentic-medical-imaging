"""NLP gold-set evaluation: entity precision/recall/F1 + orientation accuracy.

Usage: python t3_agentic/eval/eval_nlp.py
"""
import json
import os
import sys

ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

from t3_agentic.nlp.server import analyze_text

GOLD = os.path.join(os.path.dirname(__file__), "gold_notes_test.json")
OUT = os.path.join(ROOT, "outputs")


def prf(tp, fp, fn):
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f


def evaluate(path):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    tp = fp = fn = 0
    orient_ok = 0
    rows = []
    for note in data["notes"]:
        got = analyze_text(note["text"])
        gold = set(note["entities"])
        pred = set(got["entities"])
        t = len(gold & pred)
        tp += t
        fp += len(pred - gold)
        fn += len(gold - pred)
        ok = got["orientation"] == note["orientation"]
        orient_ok += ok
        rows.append({"id": note["id"], "gold": sorted(gold), "pred": sorted(pred),
                     "orientation_gold": note["orientation"],
                     "orientation_pred": got["orientation"], "orientation_ok": ok,
                     "redactions": got.get("redactions", 0)})
    p, r, f = prf(tp, fp, fn)
    summary = {
        "split": os.path.splitext(os.path.basename(path))[0],
        "n_notes": len(data["notes"]),
        "entity_precision": round(p, 4),
        "entity_recall": round(r, 4),
        "entity_f1": round(f, 4),
        "orientation_accuracy": round(orient_ok / len(data["notes"]), 4),
        "tp": tp, "fp": fp, "fn": fn,
    }
    return summary, rows


def main():
    here = os.path.dirname(__file__)
    fresh = os.environ.get("NLP_GOLD_FILE")
    dev_summary, _ = evaluate(os.path.join(here, "gold_notes_dev.json"))
    test_summary, rows = evaluate(os.path.join(here, "gold_notes_test.json"))
    if fresh:
        fresh_summary, fresh_rows = evaluate(fresh)
    else:
        fresh_summary, fresh_rows = None, []
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "nlp_eval.json"), "w", encoding="utf-8") as fh:
        json.dump({"dev": dev_summary, "test": test_summary, "test_notes": rows,
                   "fresh": fresh_summary, "fresh_notes": fresh_rows},
                  fh, ensure_ascii=False, indent=2)
    print("=== dev (utilise pour regler les regles) ===")
    print(json.dumps(dev_summary, indent=2, ensure_ascii=False))
    print("\n=== test (split held-out) ===")
    print(json.dumps(test_summary, indent=2, ensure_ascii=False))
    if fresh_summary:
        print("\n=== fresh (held-out, jamais utilise pour regler) ===")
        print(json.dumps(fresh_summary, indent=2, ensure_ascii=False))
        for row in fresh_rows:
            if not row["orientation_ok"] or row["gold"] != row["pred"]:
                print(f"  {row['id']}: gold={row['gold']} pred={row['pred']} "
                      f"| orient gold={row['orientation_gold']} pred={row['orientation_pred']}")
    print("\nmismatches (test):")
    for row in rows:
        if not row["orientation_ok"] or row["gold"] != row["pred"]:
            print(f"  {row['id']}: gold={row['gold']} pred={row['pred']} "
                  f"| orient gold={row['orientation_gold']} pred={row['orientation_pred']}")


if __name__ == "__main__":
    main()
