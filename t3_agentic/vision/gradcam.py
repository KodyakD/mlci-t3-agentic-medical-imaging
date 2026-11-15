import numpy as np

# tensorflow is lazy-imported inside the gradient functions so importing the
# vision server does not pay the TF import cost when only ONNX classify is used.
_TF = None


def _tf():
    """Import TensorFlow on first use (importing it costs ~15 s)."""
    global _TF
    if _TF is None:
        import tensorflow as tf
        _TF = tf
    return _TF


def _forward_factory(model):
    tf = _tf()
    """Build the forward pass that exposes (last conv activations, positive logit).

    The DenseNet121 backbone is a NESTED Functional model inside the outer model,
    so `model.layers` contains no Conv2D: the last conv has to be found inside
    the backbone. The conv tensor and the prediction must be produced by the SAME
    forward pass, otherwise the tape has no path from score to conv and returns
    None gradients.
    """
    base = next((l for l in model.layers if isinstance(l, tf.keras.Model)), None)
    if base is None:
        raise RuntimeError("backbone imbriqué introuvable dans le modèle")
    convs = [l for l in base.layers if isinstance(l, tf.keras.layers.Conv2D)]
    if not convs:
        raise RuntimeError("aucune couche Conv2D dans le backbone")
    last_conv = convs[-1]
    core = tf.keras.Model(base.inputs, [last_conv.output, base.output])
    norm = next((l for l in model.layers if isinstance(l, tf.keras.layers.Normalization)), None)
    if norm is None:
        raise RuntimeError("couche Normalization introuvable")
    tail = model.layers[model.layers.index(base) + 1:]

    def forward(x):
        h = norm(x, training=False)
        conv, h = core(h, training=False)
        for layer in tail:
            h = layer(h, training=False)
        return conv, h[:, 0]

    return forward


def explain(model, xb, class_idx=1):
    tf = _tf()
    """Grad-CAM on the predicted class. Returns (cam 8x8 float32, p_pneumonia)."""
    forward = _forward_factory(model)
    with tf.GradientTape() as tape:
        conv, prob = forward(tf.convert_to_tensor(xb))
        eps = tf.keras.backend.epsilon()
        logit = tf.math.log(prob + eps) - tf.math.log(1.0 - prob + eps)
        score = logit if class_idx == 1 else -logit
    grads = tape.gradient(score, conv)
    if grads is None:
        raise RuntimeError("gradient None: la branche conv n'est pas sur le chemin du score")
    w = tf.reduce_mean(grads, axis=(1, 2))
    cam = tf.nn.relu(tf.reduce_sum(conv * w[:, None, None, :], axis=-1))[0].numpy()
    cam = cam / (cam.max() + 1e-8)
    return cam, float(prob.numpy()[0])
