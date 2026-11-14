"""JSONL trace journal: tool calls, harness decisions, fusion decisions."""
import json
import os
import time

ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))
LOG = os.path.join(ROOT, "outputs", "agent_log.jsonl")


def set_log(path):
    """Point the journal at another file (one per evaluation suite) and truncate it."""
    global LOG
    LOG = path
    reset()


def _write(record):
    os.makedirs(os.path.dirname(LOG), exist_ok=True)
    record = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), **record}
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def log_tool_call(trace, tool, args, result, duration_s, forced=False, fallback=False):
    _write({
        "type": "tool_call",
        "run_id": trace.get("run_id"),
        "tool": tool,
        "args": {k: (v if isinstance(v, (int, float, bool, str)) else str(v))
                 for k, v in (args or {}).items()},
        "status": "error" if isinstance(result, dict) and "error" in result else "ok",
        "duration_s": round(duration_s, 3),
        "forced": bool(forced),
        "fallback": bool(fallback),
        "result": result,
    })


def log_fusion(trace, fusion):
    _write({"type": "fusion", "run_id": trace.get("run_id"), "fusion": fusion})


def log_run(trace):
    _write({"type": "run", **trace})


def reset():
    os.makedirs(os.path.dirname(LOG), exist_ok=True)
    open(LOG, "w", encoding="utf-8").close()
