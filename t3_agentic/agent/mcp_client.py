"""Persistent MCP client pool.

One subprocess + one ClientSession per server, opened once per agent run.
Spawning a server per call would import TensorFlow on every call (seconds) and
would make the logged call durations meaningless.

Implementation note: the stdio context managers must be ENTERED and EXITED in
the same asyncio task, otherwise anyio raises
"Attempted to exit cancel scope in a different task than it was entered in".
So one long-lived coroutine owns the stack for the whole lifetime of the pool
and only exits when asked to stop.
"""
import asyncio
import json
import os
import subprocess
import sys
import threading
import time
from contextlib import AsyncExitStack

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

SERVERS = {
    "vision": "t3_agentic.vision.server",
    "nlp": "t3_agentic.nlp.server",
    "resources": "t3_agentic.resources.server",
}


class MCPPool:
    def __init__(self, servers=None, cwd=None, timeout=120):
        self.servers = dict(servers or SERVERS)
        self.cwd = cwd or os.path.normpath(
            os.path.join(os.path.dirname(__file__), "..", ".."))
        self.loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self.loop.run_forever, daemon=True)
        self._thread.start()
        self.sessions = {}
        self.tool2srv = {}
        self._ready = asyncio.Event()
        self._stop = asyncio.Event()
        self._error = None
        self._task = asyncio.run_coroutine_threadsafe(self._session(), self.loop)
        # wait until the servers are up (the session coroutine keeps running)
        asyncio.run_coroutine_threadsafe(self._ready.wait(), self.loop).result(timeout)
        if self._error:
            raise self._error

    async def _session(self):
        try:
            async with AsyncExitStack() as stack:
                env = {**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONUTF8": "1"}
                for name, module in self.servers.items():
                    params = StdioServerParameters(command=sys.executable,
                                                   args=["-m", module],
                                                   env=env, cwd=self.cwd)
                    r, w = await stack.enter_async_context(stdio_client(params))
                    session = await stack.enter_async_context(ClientSession(r, w))
                    await session.initialize()
                    self.sessions[name] = session
                    listed = await session.list_tools()
                    for tool in listed.tools:
                        self.tool2srv[tool.name] = name
                self._ready.set()
                await self._stop.wait()
        except Exception as e:  # pragma: no cover
            self._error = e
            self._ready.set()

    def _run(self, coro, timeout=120):
        return asyncio.run_coroutine_threadsafe(coro, self.loop).result(timeout)

    def call_tool(self, tool, args, timeout=180, retries=1):
        srv = self.tool2srv.get(tool)
        if srv is None:
            raise KeyError(f"outil inconnu: {tool}")
        last = None
        for attempt in range(retries + 1):
            t0 = time.perf_counter()
            try:
                future = asyncio.run_coroutine_threadsafe(
                    self.sessions[srv].call_tool(tool, args), self.loop)
                result = future.result(timeout)
            except Exception as e:
                last = e
                time.sleep(1.0)
                continue
            duration = time.perf_counter() - t0
            payload = None
            structured = getattr(result, "structuredContent", None)
            if structured:
                payload = structured
            else:
                for item in result.content:
                    if getattr(item, "type", None) == "text":
                        try:
                            payload = json.loads(item.text)
                            break
                        except Exception:
                            payload = {"raw": item.text}
            if payload is None:
                payload = {}
            return {"result": payload, "duration_s": round(duration, 3), "server": srv}
        raise last if last else RuntimeError(f"echec appel outil {tool}")

    def read_resource(self, uri, timeout=60):
        session = self.sessions.get("resources")
        if session is None:
            raise KeyError("serveur resources indisponible")
        future = asyncio.run_coroutine_threadsafe(session.read_resource(uri), self.loop)
        res = future.result(timeout)
        return "\n".join(c.text for c in res.contents
                         if getattr(c, "type", None) == "text")

    def _child_pids(self):
        """PIDs of the stdio servers we spawned (they can outlive us on Windows)."""
        try:
            out = subprocess.run(
                ["powershell", "-NoProfile", "-Command",
                 f"(Get-CimInstance Win32_Process -Filter 'ParentProcessId={os.getpid()}')"
                 ".ProcessId"],
                capture_output=True, text=True, timeout=20).stdout
            return [int(x) for x in out.split() if x.strip().isdigit()]
        except Exception:
            return []

    def close(self, timeout=30):
        children = self._child_pids()
        try:
            self.loop.call_soon_threadsafe(self._stop.set)
            self._task.result(timeout)
        except Exception:
            pass
        try:
            self.loop.call_soon_threadsafe(self.loop.stop)
        except Exception:
            pass
        # the stdio servers are subprocesses: make sure they are gone
        for pid in children:
            try:
                subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"],
                               capture_output=True, timeout=20)
            except Exception:
                pass
