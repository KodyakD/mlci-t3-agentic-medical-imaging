"""Vision tests: preprocessing contract, and the CNN when the weights are present."""
import os
import sys

import numpy as np
import pytest

ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

from t3_agentic.vision.model import (IMG_SIZE, MODEL_PATH, preprocess_gray,
                                     to_input)

WEIGHTS_OK = os.path.exists(MODEL_PATH)


def test_preprocess_shape_and_dtype():
    g = (np.random.default_rng(0).random((512, 400)) * 255).astype(np.uint8)
    a = preprocess_gray(g)
    assert a.shape == IMG_SIZE
    assert a.dtype == np.uint8


def test_preprocess_is_deterministic():
    g = (np.random.default_rng(1).random((300, 300)) * 255).astype(np.uint8)
    assert np.array_equal(preprocess_gray(g), preprocess_gray(g))


def test_to_input_is_3ch_float():
    a = (np.random.default_rng(2).random(IMG_SIZE) * 255).astype(np.uint8)
    x = to_input(a)
    assert x.shape == (256, 256, 3)
    assert x.dtype == np.float32
    assert 0.0 <= x.min() and x.max() <= 1.0


@pytest.mark.skipif(not WEIGHTS_OK, reason="poids CNN non presents (t3_agentic/scripts/fetch_assets.py)")
def test_gradcam_runs_on_the_merged_model():
    from t3_agentic.vision.gradcam import explain
    from t3_agentic.vision.model import get_model, to_input
    model = get_model()
    xb = np.expand_dims(to_input((np.random.default_rng(3).random(IMG_SIZE) * 255)
                                  .astype(np.uint8)), 0)
    cam, p = explain(model, xb, 1)
    assert cam.ndim == 2
    assert np.isfinite(cam).all()
    assert 0.0 <= p <= 1.0
