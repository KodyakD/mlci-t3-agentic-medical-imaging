"""Process-wide runtime for the Streamlit app: ONE MCP pool, warmed once.

Streamlit re-executes the app script on every interaction, so anything kept in
the script's namespace is rebuilt each time. A pool created per rerun means:
  - the four MCP servers are spawned again on every click (seconds each), and
  - the TensorFlow warm-up runs again, several times in parallel, which
    starves the LLM server (a 1 s answer became 198 s).

This module is imported, not re-executed, so it keeps one pool per process.
`get_pool()` blocks the first caller until the pool is ready, so a click that
arrives during the warm-up waits for it instead of starting a second one.
"""
import os
import threading
import time

from t3_agentic.agent.mcp_client import MCPPool

ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))

_lock = threading.Lock()
_pool = None


def get_pool(warm=True):
    """Return the shared, warm MCP pool. Blocks until it is ready."""
    global _pool
    with _lock:
        if _pool is None:
            t0 = time.time()
            pool = MCPPool()
            print(f"[UI] pool ready in {time.time()-t0:.1f}s", flush=True)
            if warm:
                try:
                    folder = os.path.join(ROOT, "demo_images", "kermany_test", "NORMAL")
                    if os.path.isdir(folder):
                        sample = sorted(os.listdir(folder))[0]
                        t1 = time.time()
                        pool.call_tool("classify_image",
                                      {"path": os.path.join(folder, sample)})
                        pool.call_tool("explain_image",
                                      {"path": os.path.join(folder, sample)})
                        print(f"[UI] vision model warm after {time.time()-t1:.1f}s",
                              flush=True)
                except Exception as e:  # pragma: no cover
                    print(f"[UI] warm-up failed: {e}", flush=True)
            _pool = pool
        return _pool
