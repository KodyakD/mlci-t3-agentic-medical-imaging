"""Visual system for the T3 clinical workbench.

Locked palette (technical / dark, seed 13) generated with the tastemaker
palette generator and verified with its contrast matrix:

  text #e6f5fb · bg #0a1114 · surface #151d1f · primary #0681a0
  on-primary #ffffff · secondary #0e333f · accent #e28158 · border #21292c

Legal pairings (from the generator's matrix):
  text-safe  : text/bg, text/surface, bg/accent, surface/accent, accent/border,
                primary/on-primary
  UI-safe    : bg/primary, surface/primary, text/primary, primary/border
  decorative : accent/on-primary, text/accent, bg/border, surface/border
                (hairlines and tints only, never the sole carrier of state)

Type pairing: one UI sans (Inter -> Segoe UI -> system) and one mono for
values and identifiers (JetBrains Mono -> Cascadia Mono -> Consolas). The
mono/sans split is what makes a clinical console read as an instrument rather
than a document.

Icons: real Lucide SVGs in assets/icons, tinted to the primary.
"""
import os

ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))
ICON_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "icons")

TOKENS = {
    "text": "#e6f5fb",
    "bg": "#0a1114",
    "surface": "#151d1f",
    "primary": "#0681a0",
    "on_primary": "#ffffff",
    "secondary": "#0e333f",
    "accent": "#e28158",
    "border": "#21292c",
}

_ICON_CACHE = {}


def icon(name, size=18, stroke=None, css_class="t3-icon"):
    """Inline a real Lucide SVG (no emoji, no drawn-by-hand paths)."""
    key = (name, size, stroke)
    if key not in _ICON_CACHE:
        path = os.path.join(ICON_DIR, f"{name}.svg")
        svg = ""
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                svg = f.read()
        svg = (svg.replace('width="48" height="48"', f'width="{size}" height="{size}"')
                  .replace('stroke="#0681a0"', f'stroke="{stroke or TOKENS["primary"]}"'))
        _ICON_CACHE[key] = svg
    return f'<span class="{css_class}" aria-hidden="true">{_ICON_CACHE[key]}</span>'


CSS = """
<style>
/* ------------------------------------------------------------------ *
 * T3 clinical workbench - type, spacing and component system.
 * Everything here is a token from the locked palette; no ad-hoc colors.
 * ------------------------------------------------------------------ */
:root {
  --t3-text: #e6f5fb;
  --t3-bg: #0a1114;
  --t3-surface: #151d1f;
  --t3-primary: #0681a0;
  --t3-on-primary: #ffffff;
  --t3-secondary: #0e333f;
  --t3-accent: #e28158;
  --t3-border: #21292c;
  --t3-s1: 4px; --t3-s2: 8px; --t3-s3: 12px; --t3-s4: 16px;
  --t3-s5: 24px; --t3-s6: 32px; --t3-s7: 48px;
  --t3-sans: "Inter", "Segoe UI", system-ui, -apple-system, "Helvetica Neue", sans-serif;
  --t3-mono: "JetBrains Mono", "Cascadia Mono", "Consolas", ui-monospace, monospace;
}

html, body, [class*="stApp"] { background: var(--t3-bg); }
.stApp { color: var(--t3-text); font-family: var(--t3-sans); font-size: 15px; }

/* Masthead: product name on the left, the non-diagnostic warning on the
   right. One job, one line, no hero. */
.block-container { padding-top: var(--t3-s5); padding-bottom: var(--t3-s6); max-width: 1180px; }

h1, h2, h3 { font-family: var(--t3-sans); letter-spacing: -0.01em; }
h1 { font-size: 1.35rem !important; font-weight: 620 !important; margin: 0 0 var(--t3-s1) 0 !important; }
h2, h3 { font-size: 0.78rem !important; font-weight: 620 !important;
         text-transform: uppercase; letter-spacing: 0.09em;
         color: var(--t3-text); opacity: 0.62; margin: 0 0 var(--t3-s3) 0 !important; }

/* The warning is a rule, not a banner: amber hairline + amber label. */
.t3-warn {
  border: 1px solid var(--t3-accent);
  border-left-width: 3px;
  border-radius: 3px;
  background: transparent;
  padding: var(--t3-s2) var(--t3-s3);
  color: var(--t3-accent);
  font-size: 0.74rem; font-weight: 560; letter-spacing: 0.03em;
}
.t3-masthead { display: flex; align-items: baseline; justify-content: space-between;
                gap: var(--t3-s4); border-bottom: 1px solid var(--t3-border);
                padding-bottom: var(--t3-s3); margin-bottom: var(--t3-s5); }
.t3-brand { display: flex; align-items: center; gap: var(--t3-s2); }
.t3-brand .t3-icon { color: var(--t3-primary); }
.t3-kicker { font-family: var(--t3-mono); font-size: 0.68rem; letter-spacing: 0.14em;
             text-transform: uppercase; color: var(--t3-primary); }

/* Metadata rail under the masthead: the fitted operating point, read as a
   specification rather than as prose. */
.t3-rail { display: flex; flex-wrap: wrap; gap: var(--t3-s2) var(--t3-s5);
           border-bottom: 1px solid var(--t3-border); padding-bottom: var(--t3-s4);
           margin-bottom: var(--t3-s5); }
.t3-rail div { font-family: var(--t3-mono); font-size: 0.72rem; color: var(--t3-text); opacity: 0.74; }
.t3-rail b { font-weight: 600; opacity: 1; color: var(--t3-primary); }

/* Answer panel: the primary object of the screen, so it gets the surface,
   a primary rule and generous internal padding. */
.t3-panel { background: var(--t3-surface); border: 1px solid var(--t3-border);
            border-left: 3px solid var(--t3-primary); border-radius: 4px;
            padding: var(--t3-s5); }
.t3-panel p { font-size: 0.95rem; line-height: 1.62; margin: 0 0 var(--t3-s3) 0; }
.t3-panel p:last-child { margin-bottom: 0; }
.t3-answer { font-size: 1rem; line-height: 1.66; color: var(--t3-text); }

/* Verdict strip: the fusion decision as chips, not a JSON blob. */
.t3-verdict { display: flex; flex-wrap: wrap; gap: var(--t3-s2); margin-bottom: var(--t3-s3); }
.t3-chip { font-family: var(--t3-mono); font-size: 0.7rem; letter-spacing: 0.05em;
           border: 1px solid var(--t3-border); border-radius: 3px; padding: 3px var(--t3-s2);
           color: var(--t3-text); opacity: 0.82; }
.t3-chip b { font-weight: 620; opacity: 1; color: var(--t3-primary); }
.t3-chip--accent { border-color: var(--t3-accent); color: var(--t3-accent); }
.t3-chip--accent b { color: var(--t3-accent); }

/* Probability meter: shows the number, does not tell it. */
.t3-meter { height: 6px; border-radius: 3px; background: var(--t3-secondary);
             overflow: hidden; margin: var(--t3-s2) 0 0 0; }
.t3-meter i { display: block; height: 100%; background: var(--t3-primary); }
.t3-meter--accent i { background: var(--t3-accent); }
.t3-value { font-family: var(--t3-mono); font-size: 0.82rem; color: var(--t3-text); }
.t3-label { font-family: var(--t3-mono); font-size: 0.68rem; letter-spacing: 0.1em;
            text-transform: uppercase; opacity: 0.55; margin-bottom: var(--t3-s1); }

/* Journal: a timeline of what the agent actually called. */
.t3-step { display: grid; grid-template-columns: 20px 1fr auto; gap: var(--t3-s3);
            align-items: center; padding: var(--t3-s2) 0;
            border-bottom: 1px solid var(--t3-border); }
.t3-step:last-child { border-bottom: 0; }
.t3-step .t3-icon { color: var(--t3-primary); }
.t3-step--forced .t3-icon { color: var(--t3-accent); }
.t3-tool { font-family: var(--t3-mono); font-size: 0.78rem; }
.t3-dur { font-family: var(--t3-mono); font-size: 0.72rem; opacity: 0.6; }
.t3-note { font-size: 0.82rem; opacity: 0.82; line-height: 1.6; }

/* Entities found in the clinical note. */
.t3-entity { display: inline-block; font-family: var(--t3-mono); font-size: 0.74rem;
              background: var(--t3-secondary); color: var(--t3-text);
              border-radius: 3px; padding: 2px var(--t3-s2); margin: 0 var(--t2) var(--t2) 0; }
.t3-entity--denied { background: transparent; border: 1px solid var(--t3-border);
                      text-decoration: line-through; opacity: 0.62; }

/* Streamlit primitives, aligned to the same tokens. */
.stButton > button[kind="primary"] {
  background: var(--t3-primary); color: var(--t3-on-primary);
  border: 1px solid var(--t3-primary); border-radius: 3px;
  font-family: var(--t3-sans); font-weight: 560; font-size: 0.88rem;
  padding: var(--t3-s2) var(--t3-s5); min-height: 38px;
  transition: filter 140ms ease, border-color 140ms ease;
}
.stButton > button[kind="primary"]:hover { filter: brightness(1.14); color: var(--t3-on-primary); }
.stButton > button[kind="primary"]:active { filter: brightness(0.94); }
.stButton > button:focus-visible { outline: 2px solid var(--t3-accent); outline-offset: 2px; }
.stTextInput input, .stTextArea textarea, .stSelectbox [data-baseweb="select"] > div {
  background: var(--t3-surface); color: var(--t3-text);
  border-color: var(--t3-border); border-radius: 3px;
  font-family: var(--t3-sans); font-size: 0.88rem;
}
.stTextInput input:focus, .stTextArea textarea:focus {
  border-color: var(--t3-primary); outline: none;
  box-shadow: 0 0 0 1px var(--t3-primary);
}
label, .stCaption, [data-testid="stWidgetLabel"] label { color: var(--t3-text) !important; }
.stCaption, small { opacity: 0.6; font-size: 0.72rem; }
[data-testid="stSidebar"] { background: var(--t3-surface); border-right: 1px solid var(--t3-border); }
[data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 { opacity: 0.6; }
hr { border-color: var(--t3-border); }
[data-testid="stExpander"] { border-color: var(--t3-border); border-radius: 3px;
                             background: var(--t3-surface); }
[data-testid="stExpander"] summary { font-size: 0.78rem; }
[data-testid="stImage"] img { border: 1px solid var(--t3-border); border-radius: 3px; }
.stAlert { border-radius: 3px; background: var(--t3-surface);
           border: 1px solid var(--t3-accent); color: var(--t3-accent); }
[data-testid="stJson"] { background: var(--t3-surface); border: 1px solid var(--t3-border);
                         border-radius: 3px; font-family: var(--t3-mono); font-size: 0.72rem; }
.t3-log { border: 1px solid var(--t3-border); border-radius: 3px;
          background: var(--t3-surface); padding: var(--t3-s3); margin-bottom: var(--t3-s3); }
.t3-log-row { font-family: var(--t3-mono); font-size: 0.74rem; color: var(--t3-text);
               opacity: 0.82; padding: 3px 0; border-bottom: 1px solid var(--t3-border); }
.t3-log-row:last-child { border-bottom: 0; color: var(--t3-primary); }
.t3-icon { display: inline-flex; align-items: center; }
.t3-icon svg { display: block; }

@media (prefers-reduced-motion: reduce) {
  * { transition-duration: 0.01ms !important; animation-duration: 0.01ms !important; }
}
</style>
"""
