"""TF-IDF retrieval over the local reference corpus (no external service)."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from corpus import CORPUS

_model = None
_matrix = None
_docs = None


def _build():
    global _model, _matrix, _docs
    if _model is not None:
        return
    from sklearn.feature_extraction.text import TfidfVectorizer
    docs = [f"{d['title']}. {d['text']} {' '.join(d['tags'])}" for d in CORPUS]
    _model = TfidfVectorizer(lowercase=True, stop_words="english", ngram_range=(1, 2))
    _matrix = _model.fit_transform(docs)
    _docs = CORPUS


def search(query, k=3):
    """Return the k most relevant passages for a query, as a list of dicts."""
    _build()
    q = _model.transform([query or ""])
    sims = (q @ _matrix.T).toarray().ravel()
    order = np.argsort(-sims)[:k]
    hits = []
    for i in order:
        if sims[i] <= 0:
            continue
        doc = CORPUS[int(i)]
        hits.append({"id": doc["id"], "title": doc["title"], "score": round(float(sims[i]), 4),
                     "text": doc["text"], "tags": doc["tags"]})
    if not hits:  # fall back to the general guidelines so the tool is never empty
        from content import GUIDELINES_PNEUMONIA
        hits = [{"id": "guidelines", "title": "Reperes de lecture", "score": 0.0,
                 "text": GUIDELINES_PNEUMONIA, "tags": ["guidelines"]}]
    return hits
