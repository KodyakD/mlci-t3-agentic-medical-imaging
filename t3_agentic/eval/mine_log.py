"""Phase 3.9 + 3.10 : mine the journal for real, citable evidence.

- the single raw tool-choice miss (if any) and which case it is
- forced-completion and fallback rates
- for each case of the evaluation: the exact lines of outputs/agent_log.jsonl
  that justify it (so every failure case in the report can cite a real line)

Usage: python t3_agentic/eval/mine_log.py [agent_eval.json ...]
"""
import json
import os
import sys

ROOT = r"C:\Users\PC\Desktop\M2S1\MLCI"
LOG = os.path.join(ROOT, "outputs", "agent_log.jsonl")
OUT = os.path.join(ROOT, "outputs")


def load_log():
    """Read every per-suite journal (one file per evaluation suite)."""
    import glob
    lines = []
    for path in sorted(glob.glob(os.path.join(ROOT, "outputs", "agent_log_*.jsonl"))):
        suite = os.path.splitext(os.path.basename(path))[0].replace("agent_log_", "")
        with open(path, encoding="utf-8") as f:
            for n, raw in enumerate(f, 1):
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    rec = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                rec["_line"] = f"{os.path.basename(path)}:{n}"
                rec["_suite"] = suite
                lines.append(rec)
    return lines


def main(eval_files):
    lines = load_log()
    runs = {}
    for rec in lines:
        rid = rec.get("run_id")
        if not rid:
            continue
        entry = runs.setdefault(rid, {"tool_calls": [], "fusion": None, "run": None,
                                      "lines": []})
        entry["lines"].append(rec["_line"])
        if rec.get("type") == "tool_call":
            entry["tool_calls"].append({"tool": rec["tool"], "line": rec["_line"],
                                        "duration_s": rec.get("duration_s"),
                                        "forced": rec.get("forced"),
                                        "status": rec.get("status")})
        elif rec.get("type") == "fusion":
            entry["fusion"] = rec
        elif rec.get("type") == "run":
            entry["run"] = rec

    report = {"n_log_lines": len(lines), "n_runs": len(runs), "per_eval": {}}

    ordered_runs = [e for e in runs.values() if e.get("run")]
    ordered_runs.sort(key=lambda e: (e.get("run") or {}).get("ts") or "")
    case_index = {}
    for ef in eval_files:
        pth = ef if os.path.isabs(ef) else os.path.join(ROOT, ef)
        if not os.path.exists(pth):
            continue
        data_ = json.load(open(pth, encoding="utf-8"))
        for i, c_ in enumerate(data_["cases"]):
            case_index[c_["id"]] = i

    for ef in eval_files:
        path = ef if os.path.isabs(ef) else os.path.join(ROOT, ef)
        if not os.path.exists(path):
            continue
        data = json.load(open(path, encoding="utf-8"))
        name = os.path.splitext(os.path.basename(path))[0]
        misses, forced_runs, fallback_runs, llm_runs = [], 0, 0, 0
        cases = []
        for c in data["cases"]:
            if "error" in c:
                continue
            # locate the run in the journal by question
            entry = None
            # 1) by run_id when the row carries it
            rid = c.get("run_id")
            if rid and rid in runs:
                entry = runs[rid]
            # 2) by question text
            if entry is None and c.get("question"):
                for e in runs.values():
                    r = e.get("run") or {}
                    if (r.get("question") or "").strip() == c["question"].strip():
                        entry = e
                        break
            # 3) positional: the journal is written in case order
            if entry is None:
                idx = case_index.get(c["id"])
                if idx is not None and idx < len(ordered_runs):
                    entry = ordered_runs[idx]
            src = (entry or {}).get("run", {}).get("tool_choice_source")
            if src == "llm":
                llm_runs += 1
            else:
                fallback_runs += 1
            if (entry or {}).get("run", {}).get("forced"):
                forced_runs += 1
            if not c.get("tool_choice_raw", True):
                misses.append({
                    "case": c["id"], "question": c.get("question"),
                    "gold_tools": c.get("gold_tools"),
                    "proposed": c.get("proposed"), "called": c.get("called"),
                    "forced": c.get("forced"),
                    "tool_choice_source": src,
                    "log_lines": (entry or {}).get("lines", []),
                })
            cases.append({
                "case": c["id"],
                "fusion_ok": c.get("fusion_ok"),
                "fusion_coherence": c.get("fusion_coherence"),
                "refused": c.get("refused"),
                "vision": {"label": c.get("vision_label"), "score": c.get("vision_score"),
                           "suspect": c.get("vision_suspect")},
                "text_orientation": c.get("text_orientation"),
                "tool_choice_source": src,
                "log_lines": (entry or {}).get("lines", []),
            })
        n = len(cases)
        report["per_eval"][name] = {
            "n_cases": n,
            "raw_tool_choice_misses": misses,
            "n_raw_misses": len(misses),
            "tool_choice_source_counts": {"llm": llm_runs, "fallback_or_router": fallback_runs},
            "runs_with_forced_completion": forced_runs,
            "forced_completion_rate": round(forced_runs / n, 3) if n else None,
            "cases": cases,
        }

    with open(os.path.join(OUT, "log_mining.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    for name, block in report["per_eval"].items():
        print(f"--- {name}: {block['n_cases']} cas, "
              f"{block['n_raw_misses']} misses bruts, "
              f"forced {block['forced_completion_rate']}, "
              f"source {block['tool_choice_source_counts']}")
        for m in block["raw_tool_choice_misses"]:
            print("    miss:", m["case"], "gold", m["gold_tools"], "proposed", m["proposed"],
                  "forced", m["forced"], "lignes journal", m["log_lines"][:6])
    print("journaux:", "outputs/agent_log_*.jsonl", "->", report["n_log_lines"],
          "lignes,", report["n_runs"], "tours")


if __name__ == "__main__":
    files = sys.argv[1:] or ["outputs/agent_eval_dev.json",
                             "outputs/agent_eval_heldout.json",
                             "outputs/agent_eval_fresh.json"]
    main(files)
