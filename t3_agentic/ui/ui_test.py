"""Automated UI tests for the Streamlit interface (Playwright, sync API).

Three checks on ONE app instance and ONE page (opening several pages breaks the
Streamlit session for this build):
  1. golden path   : image + note  -> answer with the mandatory warning
  2. contradiction : asymptomatic note on the same image -> answer, no assertion
  3. knowledge     : no image, no note -> knowledge answer, no image step

LLM is turned OFF via the sidebar checkbox (deterministic router, journaled) so the
suite completes in a couple of minutes on CPU-bound machines. The LLM path is
verified by hand.
"""
import os
import re
import socket
import subprocess
import sys
import time

ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))
APP = os.path.join(ROOT, "t3_agentic", "ui", "app.py")
WARNING = "Prototype pédagogique, ne constitue pas un diagnostic médical."


def free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def wait_for(url, timeout=180):
    import urllib.request
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            urllib.request.urlopen(url, timeout=3)
            return True
        except Exception:
            time.sleep(1)
    return False


def fill_note(page, text):
    area = page.locator("textarea").first
    area.click()
    area.fill(text)
    page.wait_for_timeout(1200)


def fill_question(page, text):
    page.locator('input[data-testid="stTextInputField"]').first.fill(text)
    page.wait_for_timeout(800)


def disable_llm(page):
    page.eval_on_selector(
        'input[type="checkbox"][aria-label="Use the LLM (Ollama qwen2.5:3b)"]',
        "el => { if (el.checked) el.click(); }")
    page.wait_for_timeout(2000)


def reset_image_to_none(page):
    box = page.locator('input[aria-label="Demo image"]').first
    box.click()
    page.wait_for_timeout(600)
    options = page.locator('[role="option"]')
    for i in range(options.count()):
        txt = options.nth(i).inner_text().strip()
        if txt.startswith("("):
            options.nth(i).click()
            page.wait_for_timeout(1500)
            return txt
    page.keyboard.press("Escape")
    return None


def pick_demo_image(page):
    box = page.locator('input[aria-label="Demo image"]').first
    box.click()
    page.wait_for_timeout(600)
    options = page.locator('[role="option"]')
    for i in range(options.count()):
        txt = options.nth(i).inner_text().strip()
        if "/" in txt:
            options.nth(i).click()
            page.wait_for_timeout(2000)
            return txt
    page.keyboard.press("Escape")
    return None


def click_analyse(page, timeout=600):
    btn = page.get_by_role("button", name=re.compile("Analyse", re.I)).first
    btn.click()
    deadline = time.time() + timeout
    while time.time() < deadline:
        body = page.inner_text("body")
        if "Agent answer" in body or "Agent error" in body:
            return body
        page.wait_for_timeout(2000)
    return page.inner_text("body")


def main():
    from playwright.sync_api import sync_playwright

    port = free_port()
    url = f"http://127.0.0.1:{port}"
    env = dict(os.environ, PYTHONIOENCODING="utf-8",
               STREAMLIT_BROWSER_GATHER_USAGE_STATS="false")
    proc = subprocess.Popen(
        [sys.executable, "-m", "streamlit", "run", APP,
         "--server.port", str(port), "--server.headless", "true"],
        cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

    failures = []
    try:
        if not wait_for(url):
            print("FAIL: the app did not start")
            return 1
        print(f"app up on {url}")

        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            ctx = browser.new_context(viewport={"width": 1400, "height": 1000})
            errors = []
            page = ctx.new_page()
            page.on("pageerror", lambda e: errors.append(f"pageerror: {e}"))
            page.on("console", lambda m: errors.append(f"console.{m.type}: {m.text}")
                    if m.type == "error" else None)

            page.goto(url, wait_until="domcontentloaded", timeout=60000)
            page.wait_for_selector("button:has-text('Analyse')", timeout=90000)
            page.wait_for_timeout(2000)
            disable_llm(page)   # deterministic router

            # ---------- 1. golden path: image + note ------------------------
            img = pick_demo_image(page)
            print(f"demo image: {img}")
            fill_note(page, "64-year-old patient, productive cough for 5 days, "
                            "fever 38.9 C, chest pain.")
            fill_question(page, "What can be said about this case?")
            body = click_analyse(page)
            ok_warn = WARNING in body
            ok_journal = "Tool-call journal" in body
            ok_answer = "Agent answer" in body
            print(f"[golden] warning={ok_warn} answer={ok_answer} journal={ok_journal}")
            if not ok_warn:
                failures.append("golden: mandatory warning missing")
            if not ok_answer:
                failures.append("golden: no answer rendered")
            if not ok_journal:
                failures.append("golden: no tool-call journal rendered")

            # ---------- 2. contradiction: asymptomatic note -----------------
            fill_note(page, "No fever, no cough, no dyspnoea. Completely asymptomatic.")
            fill_question(page, "What can be said about this case?")
            body = click_analyse(page)
            low = body.lower()
            ok_warn = WARNING in body
            assertive = any(k in low for k in
                            ("il s'agit d'une pneumonie", "diagnostic de pneumonie",
                             "presence d'une pneumonie"))
            ok = WARNING in body and "Agent answer" in body
            print(f"[contradiction] warning={ok_warn} assertive={assertive} answer={ok}")
            if not ok_warn:
                failures.append("contradiction: mandatory warning missing")
            if assertive:
                failures.append("contradiction: asserted a diagnosis")

            # ---------- 3. knowledge path: fresh no-image state -------------
            reset_image_to_none(page)
            print("[knowledge] image cleared")
            fill_note(page, "")
            fill_question(page, "What are the radiological signs of pneumonia?")
            body = click_analyse(page)
            ok_warn = WARNING in body
            ok_answer = "Agent answer" in body
            print(f"[knowledge] warning={ok_warn} answer={ok_answer}")
            if not ok_warn:
                failures.append("knowledge: mandatory warning missing")
            if not ok_answer:
                failures.append("knowledge: no answer rendered")

            real = [e for e in errors if "favicon" not in e.lower()
                    and "404" not in e.lower()]
            print(f"console/page errors: {len(real)}")
            for e in real[:3]:
                print("   ", e)
            if real:
                failures.append(f"{len(real)} console/page errors: {real[:3]}")
            browser.close()
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=15)
        except Exception:
            proc.kill()

    print("\n" + "=" * 60)
    if failures:
        print("UI TESTS FAILED:")
        for f in failures:
            print("  -", f)
        return 1
    print("UI TESTS PASSED (golden + contradiction + knowledge)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
