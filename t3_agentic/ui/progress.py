"""Live progress for the agent run: a progress bar plus the running tool log.

Streamlit scripts are synchronous, so the agent runs in a worker thread and
pushes events into a queue; the script drains the queue and updates two
placeholders. The user sees which tool is running right now, how long each
step took and the total elapsed time.
"""
import queue
import threading
import time

from t3_agentic.agent.agent import run_agent

# Ordered stages of one run -> fraction of the bar.
STAGES = [
    ("route", 0.05),
    ("tool", 0.15),    # grows per tool call
    ("fusion", 0.75),
    ("critic", 0.85),
    ("answer", 1.0),
]

TOOL_HINT = {
    "classify_image": "reading the radiograph",
    "explain_image": "computing the Grad-CAM map",
    "analyze_text": "reading the clinical note",
    "get_guideline": "loading the reading guidelines",
    "search_guidelines": "searching the reference corpus",
}


def run_with_progress(question, image_path, note, pool, use_llm, placeholder_bar,
                      placeholder_log, timeout=900):
    """Run the agent in a thread; update the bar and the log while it works.

    Returns (result, error, timing) where timing is
    {"total_s": float, "steps": [{"label": ..., "duration_s": ..., "at_s": ...}]}
    """
    events = queue.Queue()
    state = {"result": None, "error": None, "text": "reading the request"}
    timing = {"total_s": 0.0, "steps": []}
    t0 = time.time()

    def on_event(ev):
        events.put(ev)

    def worker():
        try:
            state["result"] = run_agent(question, image_path=image_path, note=note,
                                        pool=pool, use_llm=use_llm,
                                        progress=on_event)
        except Exception as e:  # pragma: no cover
            state["error"] = e
            events.put({"stage": "error", "text": f"{type(e).__name__}: {e}"})

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()

    lines = []
    seen_tools = 0
    deadline = time.time() + timeout
    while thread.is_alive() and time.time() < deadline:
        try:
            ev = events.get(timeout=0.25)
        except queue.Empty:
            continue
        stage = ev.get("stage")
        at = time.time() - t0
        if stage == "tool":
            seen_tools += 1
            tool = ev.get("tool", "?")
            hint = TOOL_HINT.get(tool, "")
            dur = ev.get("duration_s")
            forced = " (forced)" if ev.get("forced") else ""
            label = f"{tool}{forced}"
            lines.append(f"{label} · {dur:.2f} s" if dur else label)
            timing["steps"].append({"label": label, "duration_s": dur, "at_s": at})
            state["text"] = f"{tool} · {dur:.2f} s" if dur else tool
        elif stage == "fusion":
            state["text"] = f"fusion: {ev.get('text')} · t+{at:.1f} s"
            lines.append(f"fusion → {ev.get('text')} · t+{at:.1f} s"
                         + (" (refused)" if ev.get("refused") else ""))
            timing["steps"].append({"label": "fusion", "duration_s": None, "at_s": at})
        elif stage == "critic":
            state["text"] = f"critic: {ev.get('text')} · t+{at:.1f} s"
            lines.append(f"critic → {ev.get('text')} · t+{at:.1f} s")
            timing["steps"].append({"label": "critic", "duration_s": None, "at_s": at})
        elif stage == "answer":
            state["text"] = f"answer ready · t+{at:.1f} s"
            lines.append(f"answer ready · t+{at:.1f} s")
            timing["steps"].append({"label": "answer", "duration_s": None, "at_s": at})
        elif stage == "error":
            state["text"] = "error"
            lines.append(f"error → {ev.get('text')} · t+{at:.1f} s")

        frac = 0.05 + 0.7 * min(seen_tools, 4) / 4
        placeholder_bar.progress(min(frac, 0.99),
                                 text=f"{state['text']} · elapsed {time.time()-t0:.1f} s")
        placeholder_log.markdown(
            "<div class='t3-log'>"
            + "".join(f"<div class='t3-log-row'>{l}</div>" for l in lines)
            + "</div>", unsafe_allow_html=True)

    total = time.time() - t0
    timing["total_s"] = total
    if not thread.is_alive():
        placeholder_bar.progress(1.0, text=f"done · total {total:.2f} s")
    else:
        placeholder_bar.progress(1.0, text=f"timed out after {total:.0f} s")
    return state["result"], state["error"], timing
