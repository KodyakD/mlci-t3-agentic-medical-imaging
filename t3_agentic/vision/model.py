import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
# Give TF a real thread budget: by default it uses only a fraction of the
# machine's cores, which shows up as ~13% CPU during inference. We reserve
# ~80% of the 16 logical cores (13), leaving 3 for everything else.
_TF_THREADS = os.environ.get("T3_TF_THREADS", "13")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", _TF_THREADS)
os.environ.setdefault("TF_NUM_INTEROP_THREADS", _TF_THREADS)
os.environ.setdefault("OMP_NUM_THREADS", _TF_THREADS)

import cv2
import numpy as np
import tensorflow as tf

try:
    tf.config.threading.set_intra_op_parallelism_threads(
        int(os.environ.get("T3_TF_THREADS", "13")))
    tf.config.threading.set_inter_op_parallelism_threads(
        int(os.environ.get("T3_TF_THREADS", "13")))
except Exception:
    pass

IMG_SIZE = (256, 256)
CROP = 0.05
ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))
MODEL_PATH = os.environ.get(
    "T3_WEIGHTS", os.path.join(ROOT, "weights", "densenet_merged.keras"))

_model = None
_clahe = None
_onnx = None


def get_model():
    """Load the merged Keras model once per process.

    Keras 3 saves .keras as a bundle: either a zip file or a directory holding
    config.json + metadata.json + model.weights.h5. Both are accepted here.
    """
    global _model
    if _model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"poids CNN introuvables: {MODEL_PATH} (lancer fetch_assets.py)")
        _model = tf.keras.models.load_model(MODEL_PATH)
    return _model


_ONNX_PATH = os.path.join(ROOT, "weights", "densenet_merged.onnx")


def get_onnx():
    """DirectML (RTX 3050) inference via onnxruntime, fallback CPU.

    Returns an InferenceSession or None when unavailable. TF's native-Windows
    GPU support ended at TF 2.10, so ONNX+DirectML is how we get GPU on
    Windows without moving to WSL2.
    """
    global _onnx
    if _onnx is None and os.path.exists(_ONNX_PATH):
        try:
            import onnxruntime as ort
            _onnx = ort.InferenceSession(
                _ONNX_PATH,
                providers=["DmlExecutionProvider", "CPUExecutionProvider"])
        except Exception as e:
            print("ONNX/DirectML unavailable, falling back to TF:", e)
            _onnx = False
    return _onnx or None


def get_clahe():
    global _clahe
    if _clahe is None:
        _clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return _clahe


def preprocess_gray(g):
    """THE preprocessing. Training, evaluation, external tests and Flask must all use this."""
    h, w = g.shape
    dy, dx = int(h * CROP), int(w * CROP)
    g = g[dy:h - dy, dx:w - dx]
    g = cv2.resize(g, IMG_SIZE, interpolation=cv2.INTER_AREA)
    lo, hi = np.percentile(g, (1, 99))
    g = np.clip((g.astype(np.float32) - lo) / (hi - lo + 1e-6), 0, 1)
    g = (g * 255).astype(np.uint8)
    return get_clahe().apply(g)


def preprocess_file(path, already_preprocessed=False):
    """Return the 256x256 uint8 grayscale the model expects.

    already_preprocessed=True is for images that already went through
    preprocess_gray (e.g. PNGs exported by the training notebook).
    """
    g = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if g is None:
        raise ValueError(f"image illisible: {path}")
    if already_preprocessed:
        if g.shape != IMG_SIZE:
            raise ValueError(
                f"image {path} n'est pas 256x256 ({g.shape}) mais marquée comme prétraitée")
        return g
    return preprocess_gray(g)


def to_input(a):
    """uint8 256x256 grayscale -> (256, 256, 3) float32 in [0, 1]."""
    return np.repeat(a.astype("float32")[..., None] / 255.0, 3, axis=-1)


def classify(path, already_preprocessed=False):
    """Return (label, p_pneumonia, gray_uint8). label in {'pneumonie', 'normal'}."""
    a = preprocess_file(path, already_preprocessed)
    x = np.expand_dims(to_input(a), 0).astype("float32")
    sess = get_onnx()
    if sess is not None:
        p = float(sess.run(None, {"input": x})[0].flatten()[0])
    else:
        p = float(get_model().predict(x, verbose=0)[0][0])
    label = "pneumonie" if p > 0.5 else "normal"
    return label, p, a
