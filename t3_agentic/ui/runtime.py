"""Process-wide runtime for the Streamlit app: ONE MCP pool, warmed once.

Streamlit re-executes the app script on every interaction, so anything kept in
the script's namespace is rebuilt each time. A pool created per rerun means:
  - the four MCP servers are spawned again on every click (seconds each), and
  - the TensorFlow warm-up runs again, several times in parallel, which
    starves the LLM server (a 1 s answer became 198 s).

This module is imported, not re-executed, so it keeps one pool per process.
The lock is held only while the pool is CREATED; the vision warm-up runs
outside it, so a click never queues behind the warm-up for the whole of it.

`status()` is the single source of truth for the UI: it says exactly how far
the engine got, so nobody starts an analysis on a half-loaded model.
"""
import os
import threading
import time

from t3_agentic.agent.mcp_client import MCPPool

ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))

_lock = threading.Lock()          # guards _pool creation
_warm_lock = threading.Lock()     # guards the one-time vision warm-up
_pool = None
_warmed = False
_state = {
    "pool": "starting",       # starting | ready | failed
    "pool_s": 0.0,
    "vision": "pending",      # pending | warming | warm | failed
    "vision_s": 0.0,
    "error": "",
}


def status():
    """Engine state for the UI: pool, TensorFlow/Grad-CAM warm-up, timings."""
    return dict(_state)


def ready():
    return _state["pool"] == "ready" and _state["vision"] == "warm"


def _warm_vision(pool):
    """Pull TensorFlow into the vision server and compile the Grad-CAM graph.

    Once per process. Done outside the creation lock so that a click landing
    during the warm-up does not wait for the whole thing.
    """
    global _warmed
    with _warm_lock:
        if _warmed:
            return
        _state["vision"] = "warming"
        t1 = time.time()
        try:
            folder = os.path.join(ROOT, "demo_images", "kermany_test", "NORMAL")
            if os.path.isdir(folder):
                sample = sorted(os.listdir(folder))[0]
                pool.call_tool("classify_image", {"path": os.path.join(folder, sample)})
                pool.call_tool("explain_image", {"path": os.path.join(folder, sample)})
            _state["vision"] = "warm"
            _state["vision_s"] = time.time() - t1
            print(f"[UI] vision model warm after {time.time()-t1:.1f}s", flush=True)
        except Exception as e:  # pragma: no cover
            _state["vision"] = "failed"
            _state["error"] = f"{type(e).__name__}: {e}"
            print(f"[UI] warm-up failed: {e}", flush=True)
        _warmed = True


def get_pool(warm=True):
    """Return the shared MCP pool. Blocks only until the pool exists."""
    global _pool
    with _lock:
        if _pool is None:
            t0 = time.time()
            try:
                _pool = MCPPool()
                _state["pool"] = "ready"
                _state["pool_s"] = time.time() - t0
                print(f"[UI] pool ready in {time.time()-t0:.1f}s", flush=True)
            except Exception as e:  # pragma: no cover
                _state["pool"] = "failed"
                _state["error"] = f"{type(e).__name__}: {e}"
                raise
        pool = _pool
    if warm:
        _warm_vision(pool)
    return pool
