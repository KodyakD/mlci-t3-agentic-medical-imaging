"""Streamlit interface for the T3 agentic system (reasoning + Grad-CAM + journal).

Run:
    python -m streamlit run t3_agentic/ui/app.py --server.port 8001 --server.headless true

The page holds NO business logic: it calls the same agent as the evaluation
(agent/agent.py) and renders the tool-call journal.
"""
import os
import sys

import streamlit as st

ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

from t3_agentic.agent.agent import run_agent
from t3_agentic.agent.calibration import load_config
from t3_agentic.agent.mcp_client import MCPPool

st.set_page_config(page_title="T3 - Medical image analysis agent",
                   page_icon="🩻", layout="wide")

POOL = None


def get_pool():
    global POOL
    if POOL is None:
        POOL = MCPPool()
    return POOL


def demo_images():
    base = os.path.join(ROOT, "demo_images")
    out = []
    for domain in ("kermany_test", "nih_test", "nih_eval"):
        d = os.path.join(base, domain)
        if not os.path.isdir(d):
            continue
        for cls in ("NORMAL", "PNEUMONIA"):
            cd = os.path.join(d, cls)
            if not os.path.isdir(cd):
                continue
            for f in sorted(os.listdir(cd))[:12]:
                out.append((f"{domain}/{cls}/{f}", os.path.join(cd, f)))
    return out


st.title("Agentic system for medical image analysis")
st.warning("Educational prototype. Not a medical diagnosis.")

cfg = load_config()
with st.sidebar:
    st.header("Configuration")
    st.write(f"**Fitted thresholds**: TAU_LOW = {cfg['tau_low']} - TAU_HIGH = {cfg['tau_high']}")
    st.write(f"**Temperature**: {cfg['temperature']}")
    st.caption(cfg["source"])
    use_llm = st.checkbox("Use the LLM (Ollama qwen2.5:3b)", value=True)
    st.caption("When disabled, the deterministic router is used (ablation B4).")
    demos = demo_images()
    demo_choice = st.selectbox("Demo image",
                               ["(no image)"] + [d[0] for d in demos])
    st.divider()
    st.caption("MCP servers: vision (classify_image, explain_image), "
               "nlp (analyze_text), resources (get_guideline).")

question = st.text_input("Question", "What can be said about this case?")
note = st.text_area("Clinical note (optional)", height=120,
                    placeholder="64-year-old patient, productive cough for 5 days, fever 38.9 C ...")
upload = st.file_uploader("Chest X-ray (optional)", type=["png", "jpg", "jpeg"])

run = st.button("Analyse", type="primary")

if run:
    image_path = None
    if upload is not None:
        image_path = os.path.join(ROOT, "outputs", f"upload_{upload.name}")
        os.makedirs(os.path.dirname(image_path), exist_ok=True)
        with open(image_path, "wb") as f:
            f.write(upload.getbuffer())
    else:
        for name, path in demos:
            if name == demo_choice:
                image_path = path
                break

    with st.spinner("The agent is analysing the case ..."):
        try:
            out = run_agent(question, image_path=image_path,
                            note=note if note.strip() else None,
                            pool=get_pool(), use_llm=use_llm)
        except Exception as e:
            st.error(f"Agent error: {e}")
            out = None

    if out:
        left, right = st.columns([3, 2])
        with left:
            st.subheader("Agent answer")
            st.write(out["answer"])
            if out.get("fusion"):
                st.subheader("Fusion decision")
                st.json(out["fusion"])
            if out.get("vision"):
                st.subheader("CNN result")
                st.json(out["vision"])
            if out.get("text"):
                st.subheader("Note analysis")
                st.json(out["text"])
            if out.get("resource"):
                st.subheader("MCP resource consulted")
                st.json(out["resource"])
        with right:
            trace = out.get("trace", {})
            st.subheader("Tool-call journal")
            st.write(f"tool-choice source: **{trace.get('tool_choice_source')}** | "
                     f"rounds: {trace.get('rounds')} | calls: "
                     f"{len(trace.get('tool_calls') or [])} | "
                     f"forced tools: {trace.get('forced')}")
            for call in trace.get("tool_calls", []):
                st.write(f"- `{call['tool']}` ({call.get('duration_s')} s)")
            heat = (out.get("vision") or {}).get("heatmap")
            if heat and os.path.exists(heat):
                st.subheader("Grad-CAM heatmap")
                st.image(heat, caption="Visual explanation (controlled by inside_ratio)")
                st.caption(f"inside_ratio = {(out.get('vision') or {}).get('inside_ratio')}")
            else:
                st.info("No Grad-CAM heatmap (no image, or the Grad-CAM is flagged suspect).")
            with st.expander("Raw trace"):
                st.json({k: v for k, v in trace.items() if k != "answer"})
