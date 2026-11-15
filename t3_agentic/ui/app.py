"""Streamlit interface for the T3 agentic system (reasoning + Grad-CAM + journal).

Run:
    python -m streamlit run t3_agentic/ui/app.py --server.port 8001 --server.headless true

The page holds NO business logic: it calls the same agent as the evaluation
(agent/agent.py) and renders the tool-call journal.
"""
import json
import os
import sys
import time

import streamlit as st

ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

from t3_agentic.agent.agent import run_agent
from t3_agentic.agent.calibration import load_config
from t3_agentic.agent.mcp_client import MCPPool
from t3_agentic.ui.theme import CSS, TOKENS, icon

st.set_page_config(page_title="T3 · Medical image analysis workbench",
                   page_icon=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                         "assets", "icons", "scan-line.svg"),
                   layout="wide")

st.markdown(CSS, unsafe_allow_html=True)


# The pool lives in t3_agentic.ui.runtime (an imported module), so Streamlit
# reruns cannot spawn a second one. See that module for the details.
from t3_agentic.ui.runtime import get_pool, status as engine_status, ready as engine_ready


# Warm the pool in the background as soon as the app module loads, so the
# first click never pays the cold start.
import threading as _threading


def _warm_in_background():
    try:
        get_pool()
    except Exception as _e:  # pragma: no cover
        print(f"[UI] warm-up failed: {_e}", flush=True)


if os.environ.get("T3_WARM_POOL", "1") == "1":
    _threading.Thread(target=_warm_in_background, daemon=True).start()


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


def chip(label, value, accent=False):
    cls = "t3-chip t3-chip--accent" if accent else "t3-chip"
    return f'<span class="{cls}">{label} <b>{value}</b></span>'


def meter(pct, accent=False):
    cls = "t3-meter t3-meter--accent" if accent else "t3-meter"
    return f'<div class="{cls}"><i style="width:{max(0.0, min(1.0, pct)) * 100:.1f}%"></i></div>'


def html(markup):
    st.markdown(markup, unsafe_allow_html=True)


# ----------------------------------------------------------------- masthead
cfg = load_config()
html(f"""
<div class="t3-masthead">
  <div class="t3-brand">{icon("layers", 20)}
    <span><span class="t3-kicker">T3 · medical imaging workbench</span><br>
    <span style="font-size:1.32rem;font-weight:620;letter-spacing:-0.015em">
      Agent that reads the image, reads the note, and refuses when they disagree</span></span>
  </div>
  <div class="t3-warn">Prototype pédagogique · non diagnostique</div>
</div>
<div class="t3-rail">
  <div>decision threshold <b>{cfg['label_threshold']}</b></div>
  <div>trust &ge; <b>{cfg['tau_low']}</b></div>
  <div>high confidence &ge; <b>{cfg['tau_high']}</b></div>
  <div>temperature <b>{cfg['temperature']}</b></div>
  <div>fitted on <b>Kermany validation half</b> (312 images, seed 42)</div>
</div>
""")

with st.sidebar:
    st.header("Configuration")
    st.caption(cfg["source"])
    use_llm = st.checkbox("Use the LLM (Ollama qwen2.5:3b)", value=True)
    st.caption("Off = deterministic router (ablation B4).")
    demos = demo_images()
    demo_choice = st.selectbox("Demo image",
                               ["(no image)"] + [d[0] for d in demos])
    st.divider()
    st.caption("MCP servers: vision (classify_image, explain_image), "
               "nlp (analyze_text), resources (get_guideline, search_guidelines, "
               "read in every analysed case), nih (domain shift, read only).")

@st.fragment(run_every=1.0)
def engine_panel():
    """Live engine state: MCP pool + TensorFlow/Grad-CAM warm-up.

    Nothing is analysed before this says warm, so a result can never come
    from a half-loaded model.
    """
    st_ = engine_status()
    pool_s = st_.get("pool_s", 0.0)
    vision_s = st_.get("vision_s", 0.0)
    if st_.get("pool") == "ready" and st_.get("vision") == "warm":
        html(f'<div class="t3-engine t3-engine--ready">'
             f'{icon("shield-check", 16)} engine ready · MCP pool {pool_s:.0f} s '
             f'· TensorFlow + Grad-CAM {vision_s:.0f} s</div>')
    else:
        what = ("MCP servers" if st_.get("pool") != "ready"
                else "TensorFlow + Grad-CAM")
        html(f'<div class="t3-engine">{icon("activity", 16)} loading {what} …</div>')


engine_panel()

question = st.text_input("Question", "What can be said about this case?")
note = st.text_area("Clinical note (optional)", height=110,
                    placeholder="64-year-old patient, productive cough for 5 days, fever 38.9 C ...")
upload = st.file_uploader("Chest X-ray (optional)", type=["png", "jpg", "jpeg"])

def _render_body(out, timing=None):
    timing = timing or {"total_s": 0.0, "steps": []}
    left, right = st.columns([3, 2])

    with left:
        st.subheader("Agent answer")
        html(f'<div class="t3-panel t3-answer">{out["answer"]}</div>')

        fusion = out.get("fusion") or {}
        if fusion:
            st.subheader("Fusion decision")
            refused = bool(fusion.get("refused"))
            chips = [
                chip("coherence", fusion.get("coherence", "?")),
                chip("decision", fusion.get("final") or "no conclusion", accent=refused),
                chip("confidence", fusion.get("confidence", "?"), accent=refused),
                chip("refused", "yes" if refused else "no", accent=refused),
            ]
            html('<div class="t3-verdict">' + "".join(chips) + "</div>")
            html(f'<div class="t3-note">{fusion.get("reason", "")}</div>')

        vision = out.get("vision")
        if vision:
            st.subheader("CNN result")
            score = vision.get("score")
            p_raw = vision.get("p_pneumonia")
            p_cal = vision.get("p_pneumonia_calibrated")
            thr = vision.get("label_threshold", cfg.get("label_threshold"))
            html('<div class="t3-label">Model estimate of P(pneumonia), not a clinical probability</div>')
            if p_raw is not None and p_cal is not None:
                html(f'<div class="t3-value">{p_raw:.1%} raw · {p_cal:.1%} calibrated</div>')
            threshold_text = f'{float(thr):.1%}' if thr is not None else 'unavailable'
            confidence_text = f'{float(score):.1%}' if score is not None else 'unavailable'
            html(f'<div class="t3-note">Decision threshold {threshold_text} → '
                 f'<b>model prediction: {vision.get("label", "?")}</b> · '
                 f'confidence in predicted class {confidence_text}</div>')
            html('<div class="t3-note">This binary model does not identify symptoms or describe a radiograph.</div>')
            html(meter(float(score or 0.0), accent=vision.get("suspect")))
            if vision.get("suspect"):
                html(f'<div class="t3-note" style="color:{TOKENS["accent"]}">'
                     f'Grad-CAM falls outside the thorax mask, confidence downgraded.</div>')

        text = out.get("text")
        if text:
            st.subheader("Clinical note analysis (not image symptoms)")
            ent = "".join(f'<span class="t3-entity">{e}</span>'
                          for e in (text.get("entities") or []))
            den = "".join(f'<span class="t3-entity t3-entity--denied">{e}</span>'
                          for e in (text.get("denied") or []))
            html(f'<div class="t3-label">orientation · {text.get("orientation")}</div>')
            html(f'<div>{ent}{den}</div>')

        resource = out.get("resource")
        if resource:
            payload = resource.get("result", resource)
            if isinstance(payload, list):
                body = " ".join(
                    f'<div class="t3-note"><b>{h.get("title", "")}</b> {h.get("text", "")}</div>'
                    for h in payload if isinstance(h, dict))
            else:
                body = f'<div class="t3-note">{payload}</div>'
            with st.expander("General reference (not findings from this image)"):
                html(body)

    with right:
        trace = out.get("trace", {})
        st.subheader("Tool-call journal")
        source = trace.get("tool_choice_source")
        html(f'<div class="t3-rail">'
             f'<div>tool choice <b>{source}</b></div>'
             f'<div>rounds <b>{trace.get("rounds")}</b></div>'
             f'<div>calls <b>{len(trace.get("tool_calls") or [])}</b></div>'
             f'<div>forced <b>{len(trace.get("forced") or [])}</b></div>'
             f'<div>total <b>{timing.get("total_s", 0.0):.2f} s</b></div></div>')
        steps = []
        for call in trace.get("tool_calls") or []:
            forced = call.get("tool") in (trace.get("forced") or [])
            ic = "shield-check" if forced else "activity"
            dur = call.get("duration_s")
            dur_text = f"{dur:.2f} s" if dur is not None else "\u00b7"
            steps.append(
                f'<div class="t3-step{" t3-step--forced" if forced else ""}">'
                f'{icon(ic, 16)}'
                f'<span class="t3-tool">{call.get("tool")}</span>'
                f'<span class="t3-dur">{dur_text}</span></div>')
        html("".join(steps) or '<div class="t3-note">No tool call was recorded.</div>')

        heat = (out.get("vision") or {}).get("heatmap")
        if heat and os.path.exists(heat):
            st.subheader("Grad-CAM heatmap")
            st.image(heat, caption=(
                f'Grad-CAM is a model-attention heuristic, not lesion localization or proof. '
                f'inside_ratio = {(out.get("vision") or {}).get("inside_ratio")}'))
        elif out.get("vision"):
            html('<div class="t3-note">No Grad-CAM map for this run.</div>')

        with st.expander("Raw trace"):
            st.json({k: v for k, v in trace.items() if k != "answer"})


out = None
_ready = engine_ready()
run = st.button("Analyse", type="primary", disabled=not _ready,
                help=("The engine is still loading" if not _ready else ""))

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

    _t0 = time.time()
    print(f"[UI] agent start (image={bool(image_path)}, note={bool(note)}, "
          f"llm={use_llm})", flush=True)
    bar = st.progress(0.02, text="preparing the request")
    log = st.empty()
    try:
        from t3_agentic.ui.progress import run_with_progress
        out, err, timing = run_with_progress(question, image_path,
                                             note if note.strip() else None,
                                             get_pool(), use_llm, bar, log)
        if err is not None:
            raise err
        print(f"[UI] agent done in {time.time()-_t0:.1f}s "
              f"answer_len={len((out or {}).get('answer') or '')}", flush=True)
    except Exception as e:
        import traceback
        traceback.print_exc()   # full traceback in the server log
        print(f"[UI] AGENT FAILED after {time.time()-_t0:.1f}s: "
              f"{type(e).__name__}: {e}", flush=True)
        bar.empty()
        st.error(f"Agent error: {type(e).__name__}: {e}")
        out = None
        timing = {"total_s": time.time() - _t0, "steps": []}

if out:
    _tr = time.time()
    print(f"[UI] render start (total {timing.get('total_s', 0.0):.2f}s)", flush=True)
    try:
        _render_body(out, timing)
    except Exception as _e:
        import traceback
        traceback.print_exc()
        print(f"[UI] RENDER FAILED: {type(_e).__name__}: {_e}", flush=True)
        st.error(f"Render failed: {type(_e).__name__}: {_e}")
    print(f"[UI] render done in {time.time()-_tr:.1f}s", flush=True)
elif out is None and run:
    html('<div class="t3-panel"><span class="t3-note">No result was produced for this run.</span></div>')
