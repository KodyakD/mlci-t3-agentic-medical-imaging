"""scispaCy head-to-head on the French gold notes.

Runs on Python 3.9 with scispacy, in an isolated venv, because scispaCy pins
spacy<3.8 and no wheels exist for the project's Python 3.13.

Both arms share the same labelling layer: scispaCy only proposes spans, then
the project's symptom vocabulary, negation rule and orientation rule label
them. This isolates the detector (scispaCy NER vs regex) from the clinical
logic, which scispaCy does not provide for French.

Usage (PowerShell):
  & "$env:TEMP\\opencode\\venv_scispacy\\Scripts\\python.exe" `
      t3_agentic/eval/eval_nlp_scispacy.py
"""
import json
import os
import re
import sys

ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

import spacy

from t3_agentic.nlp.scrub import scrub
from t3_agentic.nlp.terms import (SYMPTOMS, _negated, orientation,
                                  strip_accents)

OUT = os.path.join(ROOT, "outputs")
SETS = ["gold_notes_dev.json", "gold_notes_test.json",
        "gold_notes_fresh.json", "gold_notes_final.json"]


def prf(tp, fp, fn):
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    return p, r, (2 * p * r / (p + r) if (p + r) else 0.0)


def label_span(span):
    """Map one scispaCy span to a project symptom label, or None."""
    text = strip_accents(span.lower())
    for pat, label in SYMPTOMS:
        if re.search(pat, text):
            return label
    return None


def evaluate(nlp, path):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    tp = fp = fn = orient_ok = 0
    raw_spans = mapped_spans = notes_with_spans = 0
    examples = []
    for note in data["notes"]:
        cleaned, _ = scrub(note["text"])
        doc = nlp(cleaned)
        labels, denied = set(), set()
        for ent in doc.ents:
            raw_spans += 1
            if len(examples) < 8:
                examples.append({"text": ent.text, "label": ent.label_})
            label = label_span(ent.text)
            if label:
                mapped_spans += 1
                if _negated(cleaned.lower(), ent.start_char):
                    denied.add(label)
                else:
                    labels.add(label)
        if doc.ents:
            notes_with_spans += 1
        gold = set(note["entities"])
        tp += len(gold & labels)
        fp += len(labels - gold)
        fn += len(gold - labels)
        if not note["text"] or not note["text"].strip():
            pred_orient = "note vide"
        else:
            pred_orient, _ = orientation(sorted(labels), sorted(denied), cleaned)
        orient_ok += pred_orient == note["orientation"]
    p, r, f = prf(tp, fp, fn)
    return {
        "split": os.path.splitext(os.path.basename(path))[0],
        "n_notes": len(data["notes"]),
        "entity_precision": round(p, 4),
        "entity_recall": round(r, 4),
        "entity_f1": round(f, 4),
        "orientation_accuracy": round(orient_ok / len(data["notes"]), 4),
        "tp": tp, "fp": fp, "fn": fn,
        "raw_scispacy_spans": raw_spans,
        "spans_mapped_to_a_label": mapped_spans,
        "notes_with_any_span": notes_with_spans,
        "span_examples": examples,
    }


def main():
    nlp = spacy.load("en_core_sci_sm")
    results = []
    for name in SETS:
        result = evaluate(nlp, os.path.join(os.path.dirname(__file__), name))
        results.append(result)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "nlp_scispacy.json"), "w", encoding="utf-8") as fh:
        json.dump({"model": "en_core_sci_sm 0.5.4", "sets": results},
                  fh, ensure_ascii=False, indent=2)
    print("wrote outputs/nlp_scispacy.json")


if __name__ == "__main__":
    main()
