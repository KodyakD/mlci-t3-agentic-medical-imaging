"""Shared calibration/threshold configuration.

Loaded from outputs/calibration_config.json (produced by eval/fit_thresholds.py).
Fitted on the Kermany validation half only; never on NIH, never on the test half.
If the file is missing, the documented placeholders are used and the system says so.
"""
import json
import os

ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))
CONFIG_PATH = os.path.join(ROOT, "outputs", "calibration_config.json")

# documented placeholders (see docs/Rapport_T3_SysA.md §7)
TAU_HIGH_DEFAULT = 0.85
TAU_LOW_DEFAULT = 0.65
TEMPERATURE_DEFAULT = 1.0

_cache = None


def load_config(force=False):
    """Return {tau_high, tau_low, temperature, fitted, source}."""
    global _cache
    if _cache is not None and not force:
        return _cache
    cfg = {"tau_high": TAU_HIGH_DEFAULT, "tau_low": TAU_LOW_DEFAULT,
           "temperature": TEMPERATURE_DEFAULT, "fitted": False,
           "source": "placeholders (aucun ajustement sur validation)"}
    if os.path.exists(CONFIG_PATH):
        try:
            with open(CONFIG_PATH, encoding="utf-8") as f:
                data = json.load(f)
            cfg.update({
                "tau_high": float(data["tau_high"]),
                "tau_low": float(data["tau_low"]),
                "temperature": float(data["temperature"]),
                "fitted": True,
                "source": data.get("source", "fit sur validation Kermany"),
            })
        except (OSError, KeyError, ValueError):
            pass
    _cache = cfg
    return cfg


def apply_temperature(p, temperature=None):
    """Temperature scaling on the P(pneumonia) probability.

    Uses the LOG-odds (logit), so the decision p > 0.5 is preserved.
    """
    import math
    t = temperature if temperature is not None else load_config()["temperature"]
    if t == 1.0:
        return float(p)
    p = min(max(float(p), 1e-7), 1 - 1e-7)
    logit = math.log(p / (1 - p))
    z = logit / t
    return 1.0 / (1.0 + math.exp(-z))
