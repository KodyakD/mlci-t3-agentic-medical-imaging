"""Fusion ablation: image-only vs text-only vs multimodal, from the journal.

Reads outputs/agent_log.jsonl (tool results per run), recomputes the three
fusion variants for each run and aggregates how often they agree.

Usage: python t3_agentic/eval/ablation.py
"""
import json
import os
import sys
from collections import Counter

ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

from t3_agentic.agent.fusion import fuse

LOG = os.path.join(ROOT, "outputs", "agent_log.jsonl")
OUT = os.path.join(ROOT, "outputs")


def load_runs():
    runs = {}
    if not os.path.exists(LOG):
        return runs
    for line in open(LOG, encoding="utf-8"):
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if rec.get("type") != "tool_call":
            continue
        rid = rec.get("run_id")
        if not rid:
            continue
        entry = runs.setdefault(rid, {"vision": None, "text": None})
        if rec["tool"] == "classify_image" and "error" not in (rec.get("result") or {}):
            v = dict(rec["result"])
            v["suspect"] = (entry["vision"] or {}).get("suspect", False)
            entry["vision"] = v
        elif rec["tool"] == "explain_image" and "error" not in (rec.get("result") or {}):
            if entry["vision"] is not None:
                entry["vision"]["suspect"] = bool(rec["result"].get("suspect"))
                entry["vision"]["inside_ratio"] = rec["result"].get("inside_ratio")
        elif rec["tool"] == "analyze_text" and "error" not in (rec.get("result") or {}):
            entry["text"] = rec["result"]
    return runs


def main():
    runs = load_runs()
    rows = []
    agree = Counter()
    for rid, entry in runs.items():
        v, t = entry["vision"], entry["text"]
        if v is None and t is None:
            continue
        f_img = fuse(v, None)
        f_txt = fuse(None, t)
        f_both = fuse(v, t)
        modality = "both" if (v is not None and t is not None) else (
            "image" if v is not None else "text")
        rows.append({"run_id": rid, "modality": modality,
                     "image_only": f_img["final"], "text_only": f_txt["final"],
                     "multimodal": f_both["final"],
                     "coherence_both": f_both["coherence"],
                     "refused_both": f_both["refused"]})
        agree["image_vs_multimodal"] += int(f_img["final"] == f_both["final"])
        agree["text_vs_multimodal"] += int(f_txt["final"] == f_both["final"])
        agree["all_three"] += int(f_img["final"] == f_txt["final"] == f_both["final"])

    # The meaningful ablation compares the three variants on the runs where BOTH
    # modalities exist (otherwise one variant is trivially None).
    both = [r for r in rows if r["modality"] == "both"]
    nb = len(both)
    summary = {"n_runs_with_tools": len(rows), "n_runs_with_both_modalities": nb}
    if nb:
        summary.update({
            "both_image_vs_multimodal_agreement": round(
                sum(r["image_only"] == r["multimodal"] for r in both) / nb, 3),
            "both_text_vs_multimodal_agreement": round(
                sum(r["text_only"] == r["multimodal"] for r in both) / nb, 3),
            "both_all_three_agreement": round(
                sum(r["image_only"] == r["text_only"] == r["multimodal"] for r in both) / nb, 3),
            "both_refusal_rate_multimodal": round(
                sum(r["refused_both"] for r in both) / nb, 3),
        })
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "ablation.json"), "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "runs": rows}, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print("\n| run | image seule | texte seul | multimodal | coherence | refus |")
    print("|---|---|---|---|---|---|")
    for r in rows:
        print(f"| {r['run_id'][:8]} | {r['image_only']} | {r['text_only']} | "
              f"{r['multimodal']} | {r['coherence_both']} | {r['refused_both']} |")


if __name__ == "__main__":
    main()
