"""Ablation against ground truth: image-only, text-only, multimodal.

For every case in the agent suite, run the agent with ONLY the image, ONLY the
note, or BOTH, and score the fusion coherence against the case's gold list.
This is the ablation the report calls for, measured against what the suite
considers a correct decision — not against what the full pipeline happened to
do. Deterministic (no LLM) by default for reproducibility.

Usage: python t3_agentic/eval/ablation_groundtruth.py
"""
import json
import os
import sys

ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

from t3_agentic.agent.agent import run_agent
from t3_agentic.agent.mcp_client import MCPPool
from t3_agentic.eval.run_eval import resolve_image, CASES, OUT

SUITES = {
    "dev": os.path.join(os.path.dirname(CASES), "cases.json"),
    "heldout": os.path.join(os.path.dirname(CASES), "cases_heldout.json"),
    "fresh": os.path.join(os.path.dirname(CASES), "cases_fresh.json"),
}


def load_cases():
    for suite, path in SUITES.items():
        with open(path, encoding="utf-8") as f:
            for c in json.load(f)["cases"]:
                yield suite, c


def main():
    pool = MCPPool()
    try:
        results = []
        n_img = n_note = 0
        hit = {"image_only": 0, "text_only": 0, "both": 0}
        tot = {"image_only": 0, "text_only": 0, "both": 0}
        for suite, case in load_cases():
            gold = case["gold_coherence"]
            gold_list = gold if isinstance(gold, list) else [gold]
            image_path = resolve_image(case.get("image_class"), case.get("image_index", 0),
                                       case.get("domain", "kermany"))
            note = case.get("note")
            row = {"id": case["id"], "suite": suite, "gold": gold_list}

            if image_path:
                out = run_agent(case["ask"], image_path=image_path, note=None,
                                pool=pool, use_llm=False)
                ok = out["fusion"]["coherence"] in gold_list
                row["image_only"] = out["fusion"]["coherence"]
                hit["image_only"] += ok
                tot["image_only"] += 1
                n_img += 1
            if note:
                out = run_agent(case["ask"], image_path=None, note=note,
                                pool=pool, use_llm=False)
                ok = out["fusion"]["coherence"] in gold_list
                row["text_only"] = out["fusion"]["coherence"]
                hit["text_only"] += ok
                tot["text_only"] += 1
                n_note += 1
            if image_path and note:
                out = run_agent(case["ask"], image_path=image_path, note=note,
                                pool=pool, use_llm=False)
                ok = out["fusion"]["coherence"] in gold_list
                row["both"] = out["fusion"]["coherence"]
                hit["both"] += ok
                tot["both"] += 1
            results.append(row)

            print(f"{suite:8s} {case['id']:4s} img_ok={hit['image_only']}/{tot['image_only']} "
                  f"note_ok={hit['text_only']}/{tot['text_only']} "
                  f"both_ok={hit['both']}/{tot['both']}", flush=True)
    finally:
        try:
            pool.close()
        except Exception:
            pass

    def rate(k):
        return round(hit[k] / tot[k], 3) if tot[k] else None

    summary = {"suites": list(SUITES), "n_cases": len(results),
               "image_only_accuracy": rate("image_only"), "n_image_cases": tot["image_only"],
               "text_only_accuracy": rate("text_only"), "n_note_cases": tot["text_only"],
               "both_accuracy": rate("both"), "n_both_cases": tot["both"]}
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "ablation_groundtruth.json"), "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "cases": results}, f,
                  ensure_ascii=False, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
