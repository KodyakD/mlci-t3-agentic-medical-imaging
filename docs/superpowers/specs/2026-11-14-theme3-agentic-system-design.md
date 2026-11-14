# Thème 3 — Système agentique d'aide à l'analyse d'images médicales : Design (v2)

Date : 2026-11-14 (v1) → 2026-11-14 (v2, après revue externe)
Statut : approuvé et **en cours d'implémentation**

## 1. Objectif

Construire le système agentique du TP « T3 : Système agentique d'aide à l'analyse d'images
médicales » : un agent piloté par un LLM local qui choisit des outils MCP, fusionne leurs
résultats, refuse de conclure quand la confiance est insuffisante ou que les modalités se
contredisent, journalise chaque appel et s'évalue.

Le composant vision s'appuie sur le modèle déjà entraîné et validé sur Kaggle
(`kermany-nih-merged` : DenseNet121, Kermany + NIH, splits par patient).

## 2. Décisions actées

| Sujet | Décision |
|---|---|
| Périmètre | Système Thème 3 complet (serveurs MCP + agent + évaluation + rapport) |
| LLM de l'agent | Ollama local, `qwen2.5:3b` (`OLLAMA_MODEL`) |
| NLP | Règles + expressions régulières (scispaCy non installé : mesuré, pas aspirationnel) |
| Données d'évaluation | Images Kermany/NIH + notes synthétiques écrites à la main |
| Poids CNN | Sortie du kernel Kaggle `abdelraoufgheyat/kermany-nih-merged` (pas de ré-entraînement) |
| Cadre agent | Boucle écrite à la main — **pas** de LangChain/CrewAI/smolagents |
| Thresholds | `TAU_HIGH=0.85`, `TAU_LOW=0.65` : **placeholders à calibrer sur le split validation** |

## 3. Résultats de référence (Kaggle, kernel `kermany-nih-merged` v1)

- Kermany test (940 images, split patient) : accuracy **0.9660**, AUC **0.9893**
  (NORMAL P 0.958 / R 0.923 ; PNEUMONIA P 0.969 / R 0.984).
- NIH held-out test (632 images) : accuracy **0.7199**, AUC **0.7436**
  (NORMAL P 0.732 / R 0.933 ; PNEUMONIA P 0.633 / R 0.253).
- Ligne de base Kermany-only (v4) : Kermany accuracy 0.843 / AUC 0.9599 ; NIH externe
  40+40 accuracy 0.57 (recall normale 0.17) — **non comparable** (split et taille différents).
- Grad-CAM : validé en local (backbone DenseNet **imbriqué**, la dernière Conv2D est dans le
  backbone ; la branche conv et la prédiction doivent être produites par la **même** passe
  forward, sinon `tape.gradient` renvoie `None`).

Point de vigilance (issu de la revue) : sur NIH, la classe majoritaire « normal » atteint déjà
**0.687** d'accuracy ; l'accuracy seule est trompeuse, l'**accuracy équilibrée** vaut ~0.593.
Le seuil doit être calibré sur la validation, pas sur le test.

## 4. Architecture (v2)

```
MLCI/t3_agentic/
  vision/     model.py (prétraitement + modèle chargé une fois) | gradcam.py | roi.py | server.py
  nlp/        terms.py (lexique + négation) | scrub.py (mentions diagnostiques) | server.py
  resources/  content.py (guidelines, ethics, exemples) | server.py (ressources + prompt)
  agent/      prompts.py | gate.py | mcp_client.py | fusion.py | logger.py | agent.py
  eval/       cases.json (20 cas agent) | gold_notes_dev.json | gold_notes_test.json
              run_eval.py | eval_nlp.py | calibration.py
  scripts/    fetch_assets.py | dump_preds.py
  tests/      pytest (gate, fusion, nlp, ressources, vision)
  README.md
```

### Principes (issus de la revue externe)

1. **Le LLM ne décide jamais d'un label, d'un score ou d'un refus.** Il propose les
   *outils* ; le code injecte les arguments, exécute, fusionne, refuse.
2. **Gate modal déterministe** : `allowed_tools(image, note)` — sans image, le CNN n'est
   jamais exposé ni appelé (exigence explicite du PDF).
3. **Clients MCP persistants** : un sous-processus + une `ClientSession` par serveur, ouverts
   une fois par exécution (sinon import TensorFlow à chaque appel et durées de logged
   appels sans sens). Le contexte stdio est ouvert **et** fermé dans la **même** tâche
   asyncio (sinon `anyio` : « Attempted to exit cancel scope in a different task »).
4. **Fusion par règles** (`agent/fusion.py`, fonctions pures) : `concordant` /
   `contradictoire` / `insuffisant` / `image_seule` / `texte_seule`, avec `final`,
   `reason`, `confidence`, `refused`.
5. **Provenance** : tout nombre de la réponse doit venir d'un résultat d'outil, sinon on
   retombe sur la réponse déterministe et on journalise la violation.
6. **Journal JSONL** (`outputs/agent_log.jsonl`) : appels d'outils (args, durée, `forced`,
   `fallback`), décisions de fusion, trace complète par tour.
7. **Avertissement non diagnostique** ajouté par le code, pas par le LLM.
8. **Calibration** : ECE + courbe de fiabilité sur les probabilités sauvegardées ; la
   température est ajustée sur la validation uniquement.

## 5. Évaluation

- **Agent** : 20 cas (`eval/cases.json`) — image seule, note seule, les deux concordants,
  contradiction, note insuffisante, injection. Score : choix d'outils (brut **et** après
  complétion du harnais), exactitude de fusion, refus, hallucinations, taux de warning,
  nombre moyen d'appels.
- **NLP** : 60 notes manuscrites, **splittées** : `gold_notes_dev.json` (utilisé pour régler
  les règles) et `gold_notes_test.json` (held-out, non utilisé pour le réglage).
  Résultat actuel : dev F1 1.000 ; **test F1 0.965** (P 1.000 / R 0.932), orientation 0.967.
  Limites assumées : 3 entités manquées (« respiration rapide », « sibilants », « shivering »)
  — le test n'est **pas** retouché après coup.
- **Vision** : matrices de confusion + AUC par domaine (chiffres Kaggle) ; ECE + reliability
  curve sur les prédictions locales (`scripts/dump_preds.py` → `eval/calibration.py`).
- **Analyse d'erreurs** : ≥ 5 cas commentés, **minés dans les logs**, jamais supposés.

## 6. Rapport T3

Sections : contexte et questions ; architecture agentique et rôle de chaque serveur MCP ;
modèle de vision (Kermany+NIH, domain shift) ; NLP (nettoyage, lexic, gold set) ; fusion
+ ablation (image seule / texte seule / multimodal) ; évaluation de l'agent ; calibration ;
analyse d'erreurs ≥ 5 cas ; éthique et limites ; conclusion.

## 7. Risques et limites

- Seuil de fusion non calibré (`TAU_HIGH`/`TAU_LOW` placeholders) — à fixer sur validation.
- NIH réduit à binaire pneumonia-like vs No Finding ; `Infiltration` n'est pas une pneumonie ;
  labels NIH extraits par NLP.
- Décalage de domaine confondant population (pédiatrique vs adulte) et acquisition.
- Split Kermany non officiel (940 vs 624) : ne pas comparer aux chiffres publiés.
- Aucune validation clinique ; l'avertissement permanent est obligatoire.
- Le jeu de notes test et les règles ont le même auteur : l'optimum reste possible ;
  un jeu tiers (par le coéquipier) est la seule mesure non biaisée.

## 8. Hors périmètre

Multi-label NIH, RAG, agent critique, interface, robustesse d'image, RSNA/CheXpert
(licences), ré-entraînement.
