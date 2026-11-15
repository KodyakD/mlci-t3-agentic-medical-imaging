"""MC-Dropout uncertainty for the merged DenseNet121 (bonus: "incertitude avancée").

The model has a Dropout(0.3) layer before the dense head, so keeping that layer
active at inference gives T stochastic predictions per image. We report:

  - mean predictive probability and its std (epistemic spread)
  - predictive entropy (mean over the T samples)
  - disagreement rate (how often the T samples disagree on the label)
  - accuracy / AUC / ECE of the MC mean (ensemble) vs a single deterministic pass
  - confidence-weighted coverage: how often the mean confidence is high while the
    samples disagree (i.e. where the model is *not* actually sure)

Usage (local):
  python t3_agentic/eval/mc_dropout.py --dir demo_images/kermany_test --out outputs/mc_kermany.npz
  python t3_agentic/eval/mc_dropout.py --dir demo_images/nih_test --out outputs/mc_nih.npz --mc 20
"""
import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

from t3_agentic.agent.calibration import apply_temperature
from t3_agentic.eval.calibration import ece
from t3_agentic.vision.model import get_model, preprocess_file, to_input


def enable_mc_dropout(model):
    """Activate ONLY the dropout layers at inference (BatchNorm stays in eval mode).

    Setting `layer.trainable = True` is not enough in Keras 3: `model.predict()`
    calls the model with training=False, which propagates to every layer. We
    therefore replace the dropout call by an inverted-dropout mask that is
    always applied, and leave BatchNorm untouched.
    """
    import tensorflow as tf
    n = 0
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.Dropout):
            rate = float(layer.rate)

            def mc_call(inputs, training=None, _rate=rate, **kwargs):
                keep = 1.0 - _rate
                mask = tf.cast(tf.random.uniform(tf.shape(inputs)) < keep, inputs.dtype)
                return inputs * mask / keep

            layer.call = mc_call
            n += 1
    print(f"MC-Dropout enabled on {n} layer(s)")
    return model


def predict_mc(model, batch, n_mc):
    """Return (T, n) probabilities and (n,) mean label probabilities."""
    outs = []
    for _ in range(n_mc):
        outs.append(model.predict(batch, verbose=0).ravel())
    return np.stack(outs, axis=0)


def collect(dirpath, limit_per_class=None):
    files, labels = [], []
    for cls in sorted(os.listdir(dirpath)):
        d = os.path.join(dirpath, cls)
        if not os.path.isdir(d):
            continue
        names = sorted(f for f in os.listdir(d) if f.lower().endswith((".png", ".jpg", ".jpeg")))
        if limit_per_class:
            names = names[:limit_per_class]
        for f in names:
            files.append(os.path.join(d, f))
            labels.append(1 if cls.upper() == "PNEUMONIA" else 0)
    return files, np.array(labels)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--mc", type=int, default=10)
    ap.add_argument("--limit-per-class", type=int, default=0)
    ap.add_argument("--files", default=None,
                    help="json list of image paths (e.g. outputs/kaggle_test_files.json)")
    ap.add_argument("--batch", type=int, default=16)
    args = ap.parse_args()

    from sklearn.metrics import roc_auc_score
    if args.files:
        import json as _json
        with open(args.files, encoding="utf-8") as fh:
            paths = _json.load(fh)
        files = paths
        y = np.array([1 if "PNEUMONIA" in p else 0 for p in paths])
    else:
        files, y = collect(args.dir, args.limit_per_class or None)
    print(f"{len(files)} images from {args.dir}", flush=True)
    if len(files) < 20:
        print("ABORT: not enough images")
        return

    model = enable_mc_dropout(get_model())
    means, stds, ents, disagreements, single = [], [], [], [], []
    for i in range(0, len(files), args.batch):
        chunk = files[i:i + args.batch]
        try:
            batch = np.stack([to_input(preprocess_file(f)) for f in chunk])
        except Exception as e:
            print("skip", chunk[0], e)
            continue
        probs = predict_mc(model, batch, args.mc)
        means.append(probs.mean(axis=0))
        stds.append(probs.std(axis=0))
        ents.append(-(probs * np.log(np.clip(probs, 1e-7, 1)) +
                     (1 - probs) * np.log(np.clip(1 - probs, 1e-7, 1))).mean(axis=0))
        lab = (probs >= 0.5).astype(int)
        disagreements.append((lab != lab[0]).any(axis=0).astype(int))
        single.append(probs[0])
        if (i + args.batch) % (args.batch * 5) == 0:
            print(f"{i + args.batch}/{len(files)}", flush=True)

    p_mc = np.concatenate(means)
    sd = np.concatenate(stds)
    ent = np.concatenate(ents)
    dis = np.concatenate(disagreements)
    p_single = np.concatenate(single)
    np.savez(args.out, p=p_mc, y=y, std=sd, entropy=ent, disagree=dis, p_single=p_single)

    def stats(p, threshold=None):
        """Metrics at a decision threshold (None -> 0.5, the un-fitted default)."""
        t = 0.5 if threshold is None else threshold
        pred = (p >= t).astype(int)
        e, _ = ece(p, y, threshold=t)
        try:
            a = roc_auc_score(y, p)
        except ValueError:
            a = float("nan")
        return {"threshold": round(float(t), 4),
                "accuracy": round(float((pred == y).mean()), 4),
                "auc": round(float(a), 4),
                "ece": round(float(e), 4),
                "predicted_positive_rate": round(float(pred.mean()), 4),
                "mean_confidence": round(float(np.where(p >= t, p, 1 - p).mean()), 4)}

    # The MC mean is its own estimator, so it gets its OWN threshold, fitted
    # with the same protocol as the main model (stratified half, seed 42,
    # fitted on the validation half only, reported on the test half).
    from sklearn.model_selection import train_test_split
    from t3_agentic.eval.fit_thresholds import fit_label_threshold
    idx = np.arange(len(p_mc))
    val_idx, test_idx = train_test_split(idx, test_size=0.5, stratify=y,
                                         random_state=42)
    mc_threshold = fit_label_threshold(p_mc[val_idx], y[val_idx])
    mc_pred = (p_mc >= mc_threshold).astype(int)

    report = {
        "dir": args.dir, "n": int(len(p_mc)), "mc_passes": args.mc,
        "threshold_fitted_on_validation_half": round(float(mc_threshold), 4),
        "single_pass": stats(p_single),
        "single_pass_at_fitted_threshold": stats(p_single, mc_threshold),
        "mc_mean": stats(p_mc),
        "mc_mean_at_fitted_threshold": stats(p_mc, mc_threshold),
        "mc_mean_test_half_at_fitted_threshold": {
            "n": int(len(test_idx)),
            "accuracy": round(float((mc_pred[test_idx] == y[test_idx]).mean()), 4),
            "predicted_positive_rate": round(float(mc_pred[test_idx].mean()), 4),
        },
        "epistemic_std_mean": round(float(sd.mean()), 4),
        "epistemic_std_p90": round(float(np.percentile(sd, 90)), 4),
        "predictive_entropy_mean": round(float(ent.mean()), 4),
        "disagreement_rate": round(float(dis.mean()), 4),
        "cases_mc_unsure_but_confident": int(((dis == 1) & (np.where(p_mc >= .5, p_mc, 1 - p_mc) >= 0.9)).sum()),
    }
    with open(args.out.replace(".npz", "_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
