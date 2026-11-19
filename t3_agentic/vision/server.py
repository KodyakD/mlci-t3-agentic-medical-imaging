"""MCP vision server: classify_image + explain_image.

stdout is reserved for the MCP protocol: nothing here may print to stdout.
"""
import os
import sys

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageFilter
from mcp.server.fastmcp import FastMCP

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gradcam import explain
from model import classify, get_model, preprocess_file, to_input
from roi import chest_roi

sys.path.insert(0, os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..")))
from t3_agentic.agent.calibration import decide, load_config

OUT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    "..", "..", "outputs"))
os.makedirs(OUT, exist_ok=True)

mcp = FastMCP("vision")


def _not_a_real_image(path):
    return not os.path.exists(path)


@mcp.tool()
def classify_image(path: str, patient_id: str | None = None,
                   already_preprocessed: bool = False) -> dict:
    """Classifie une radiographie thoracique (pneumonie vs normal).

    path: chemin local de l'image. already_preprocessed: True si l'image a deja
    passe par preprocess_gray (PNG exportes par le notebook d'entrainement).
    """
    if _not_a_real_image(path):
        return {"error": f"image introuvable: {path}"}
    try:
        p, _ = classify(path, already_preprocessed)
    except (ValueError, FileNotFoundError) as e:
        return {"error": str(e)}
    cfg = load_config()
    label, p_cal, conf = decide(p, cfg)
    return {
        "label": label,
        "p_pneumonia": round(p, 4),
        "p_pneumonia_calibrated": round(p_cal, 4),
        "score": round(conf, 4),
        "label_threshold": cfg["label_threshold"],
        "temperature": cfg["temperature"],
        "calibrated": bool(cfg["fitted"]),
        "patient": patient_id,
        "model": "densenet_merged_kermany_nih",
    }


@mcp.tool()
def explain_image(path: str, already_preprocessed: bool = False) -> dict:
    """Return a Grad-CAM visualization and its geometric thorax-mask overlap.

    The overlap is descriptive only; it is not a reliability score or patient
    finding.
    """
    if _not_a_real_image(path):
        return {"error": f"image introuvable: {path}"}
    try:
        a = preprocess_file(path, already_preprocessed)
    except (ValueError, FileNotFoundError) as e:
        return {"error": str(e)}
    model = get_model()
    xb = np.expand_dims(to_input(a), 0).astype("float32")
    p = float(model.predict(xb, verbose=0)[0][0])
    label, _, _ = decide(p)
    idx = 1 if label == "pneumonie" else 0
    cam, _ = explain(model, xb, idx)
    cam_grid = list(cam.shape)
    cam_img = Image.fromarray(np.uint8(cam / (cam.max() + 1e-8) * 255))
    cam_img = cam_img.resize((256, 256), Image.BICUBIC).filter(ImageFilter.GaussianBlur(4))
    heat = np.asarray(cam_img, dtype=np.float32) / 255.0
    gray = a.astype("float32") / 255.0
    roi = chest_roi(gray)
    inside = float((heat * roi).sum() / (heat.sum() + 1e-8))
    jet = plt.get_cmap("jet")
    base = np.stack([gray] * 3, -1)

    def overlay(cam_map):
        alpha = (0.65 * cam_map)[..., None]
        return np.clip(base * (1 - alpha) + jet(cam_map)[..., :3] * alpha, 0, 1)

    stem = os.path.splitext(os.path.basename(path))[0]
    out = os.path.join(OUT, f"gradcam_{stem}.png")
    raw_out = os.path.join(OUT, f"gradcam_raw_{stem}.png")
    plt.imsave(raw_out, overlay(heat))
    plt.imsave(out, overlay(heat * roi))
    return {
        "label": label,
        "p_pneumonia": round(p, 4),
        "heatmap": out,
        "heatmap_raw": raw_out,
        "cam_grid": cam_grid,
        "inside_ratio": round(inside, 3),
    }


def _warm_up():
    """Warm this server in the background, before anyone calls a tool.

    Two costs used to land on the click path: `import onnxruntime` (~20 s)
    and the TensorFlow import + Grad-CAM graph build (~20 s). Both happen
    here, in a thread, while the MCP handshake stays fast.
    """
    try:
        import time
        t0 = time.time()
        from model import warm_onnx, get_model
        warm_onnx()
        t1 = time.time()
        print(f"onnx session ready in {t1-t0:.1f}s", file=sys.stderr, flush=True)
        model = get_model()
        t2 = time.time()
        print(f"tensorflow + keras ready in {t2-t1:.1f}s", file=sys.stderr, flush=True)
        import numpy as np
        from gradcam import explain
        explain(model, np.zeros((1, 256, 256, 3), dtype="float32"))
        print(f"grad-cam graph compiled in {time.time()-t2:.1f}s", file=sys.stderr,
              flush=True)
    except Exception as e:  # pragma: no cover
        print(f"warm-up failed: {e}", file=sys.stderr, flush=True)


def _warm_onnx():
    """Load the ONNX session in a background thread.

    onnxruntime imports in ~19 s but does NOT suffer the Windows thread
    pathology that TensorFlow has, so it overlaps with the MCP handshake
    instead of delaying it.
    """
    try:
        import time
        t0 = time.time()
        from model import warm_onnx
        warm_onnx()
        print(f"onnx session ready in {time.time()-t0:.1f}s", file=sys.stderr, flush=True)
    except Exception as e:  # pragma: no cover
        print(f"onnx warm-up failed: {e}", file=sys.stderr, flush=True)


if __name__ == "__main__":
    import threading
    # TensorFlow: MAIN thread only. Importing it from a worker thread took
    # 105 minutes on Windows (3.8 s here), so it is imported at module level
    # and this call only loads the weights and builds the Grad-CAM graph.
    _warm_up()
    # onnxruntime: background thread, started AFTER the main thread has
    # finished its imports. Python's import lock is global, so a background
    # import started too early blocks the main thread's own imports (+20 s).
    threading.Thread(target=_warm_onnx, daemon=True).start()
    mcp.run(transport="stdio")
