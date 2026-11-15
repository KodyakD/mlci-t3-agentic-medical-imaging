"""Live progress for the agent run: a progress bar plus the running tool log.

Streamlit scripts are synchronous, so the agent runs in a worker thread and
pushes events into a queue; the script drains the queue and updates two
placeholders. The user sees which tool is being called right now, and how far
along the run is.
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

    Returns the agent result dict (or None on timeout/error).
    """
    events = queue.Queue()
    state = {"result": None, "error": None, "tools": [], "stage": "route",
             "text": "reading the request"}

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
        if stage == "tool":
            seen_tools += 1
            tool = ev.get("tool", "?")
            hint = TOOL_HINT.get(tool, "")
            dur = ev.get("duration_s")
            forced = " (forced)" if ev.get("forced") else ""
            lines.append(f"{tool}{forced}"
                         + (f" · {dur} s" if dur else "")
                         + (f" · {hint}" if hint else ""))
            state["tools"].append(tool)
            state["text"] = f"{tool}: {hint}"
        elif stage == "fusion":
            state["text"] = f"fusion decision: {ev.get('text')}"
            lines.append(f"fusion → {ev.get('text')}"
                         + (" (refused)" if ev.get("refused") else ""))
        elif stage == "critic":
            state["text"] = f"critic: {ev.get('text')}"
            lines.append(f"critic → {ev.get('text')}")
        elif stage == "answer":
            state["text"] = "answer ready"
            lines.append("answer ready")
        elif stage == "error":
            state["text"] = "error"
            lines.append(f"error → {ev.get('text')}")

        frac = 0.05 + 0.7 * min(seen_tools, 4) / 4
        for name, value in STAGES:
            if name == state.get("last_stage"):
                frac = value
        placeholder_bar.progress(min(frac, 0.99), text=state["text"])
        placeholder_log.markdown(
            "<div class='t3-log'>"
            + "".join(f"<div class='t3-log-row'>{l}</div>" for l in lines)
            + "</div>", unsafe_allow_html=True)

    if not thread.is_alive():
        placeholder_bar.progress(1.0, text=state["text"])
    return state["result"], state["error"]
