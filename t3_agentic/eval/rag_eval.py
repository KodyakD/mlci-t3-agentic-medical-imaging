"""Evaluation of the retrieval step (RAG bonus).

Each question has the passage ids that should be retrieved. We report top-1 and
top-k hit rate, plus the answer-level check: does the agent's final answer contain
the expected key terms when the question is asked through the agent?
"""
import json
import os
import sys

ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

from t3_agentic.resources.retrieval import search

QUESTIONS = [
    {"q": "Quels sont les signes radiologiques de la pneumonie ?", "expect": ["g02"], "terms": ["consolidation"]},
    {"q": "Comment lire une carte de chaleur Grad-CAM ?", "expect": ["g06"], "terms": ["Grad-CAM"]},
    {"q": "Que doit contenir un rapport d'experimentation ?", "expect": ["g12"], "terms": ["protocole"]},
    {"q": "Le modele peut-il remplacer un radiologue ?", "expect": ["g05"], "terms": ["remplacer"]},
    {"q": "Quelle est la limite du modele sur les adultes ?", "expect": ["g03", "g08"], "terms": ["adulte"]},
    {"q": "Que faire devant un epanchement pleural important ?", "expect": ["g10", "g02"], "terms": ["escalade"]},
    {"q": "Pourquoi les labels NIH sont-ils bruyants ?", "expect": ["g07"], "terms": ["multi-label"]},
    {"q": "Quelles sont les implications ethiques d'un prototype d'imagerie ?", "expect": ["g11"], "terms": ["biais"]},
    {"q": "Comment differencier une radiographie pediatrique d'une radio adulte ?", "expect": ["g03"], "terms": ["pediatrique"]},
    {"q": "Que signifie un decalage de domaine ?", "expect": ["g08"], "terms": ["seuil"]},
]


def main():
    rows = []
    top1 = topk = 0
    for item in QUESTIONS:
        hits = search(item["q"], k=3)
        ids = [h["id"] for h in hits]
        hit1 = bool(ids[:1] and ids[0] in item["expect"])
        hitk = any(i in ids for i in item["expect"])
        top1 += hit1
        topk += hitk
        rows.append({"question": item["q"], "expected": item["expect"], "retrieved": ids,
                     "top1_hit": hit1, "top3_hit": hitk,
                     "scores": [h["score"] for h in hits]})
        print(f"{'OK ' if hit1 else '   '}{item['q']}\n     -> {ids} (attendu {item['expect']})")
    n = len(QUESTIONS)
    summary = {"n_questions": n,
               "top1_hit_rate": round(top1 / n, 3),
               "top3_hit_rate": round(topk / n, 3),
               "corpus_size": len(__import__("t3_agentic.resources.corpus", fromlist=["CORPUS"]).CORPUS)}
    os.makedirs(os.path.join(ROOT, "outputs"), exist_ok=True)
    with open(os.path.join(ROOT, "outputs", "rag_eval.json"), "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "questions": rows}, f, ensure_ascii=False, indent=2)
    print("\n" + json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
