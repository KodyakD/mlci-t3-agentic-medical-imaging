# T3 — Système agentique d'aide à l'analyse d'images médicales (TP)

**Avertissement permanent :** prototype pédagogique, ne constitue pas un diagnostic médical.

Ce dépôt contient l'implémentation du système agentique du Thème 3 : un agent LLM local
qui choisit des outils exposés via MCP, fusionne leurs résultats, refuse de conclure quand
les modalités se contredisent, journalise chaque appel et s'évalue sur un jeu de cas.

## Architecture

```
t3_agentic/
  vision/      serveur MCP vision  -> classify_image, explain_image (DenseNet121 fusionné Kermany+NIH)
  nlp/         serveur MCP NLP      -> analyze_text (règles symptômes + négation + scrub diagnostique)
  resources/   serveur MCP ressources -> guidelines://pneumonia, ethics://prototype-warning, prompts
  nih/         serveur MCP NIH (second dataset, read-only) -> nih_domain_shift / nih_findings / nih_case_lookup
  agent/       boucle agent : gate déterministe -> choix d'outils LLM -> fusion -> garde-fous -> journal
  eval/        100 cas d'évaluation agent (dev 22 + heldout 18 + fresh 10 + extra 50) + calibration (ECE, courbe de fiabilité) + MC-Dropout + RAG + ablation vs ground truth
  scripts/     récupération des poids Kaggle, dump des probabilités
  tests/       pytest (gate, fusion, NLP, ressources, vision)
```

- **MCP** : transport stdio, `mcp.server.fastmcp.FastMCP` (SDK `mcp` 1.28).
- **LLM** : Ollama local, `qwen2.5:3b` par défaut (`OLLAMA_MODEL`).
- **Vision** : DenseNet121 (sans top), entraîné sur Kermany + NIH (kernel Kaggle
  `abdelraoufgheyat/kermany-nih-merged`), prétraitement identique à l'entraînement
  (crop 5 % -> resize 256 INTER_AREA -> clip percentiles 1–99 -> CLAHE -> 3 canaux -> /255).
- **Grad-CAM** : le backbone DenseNet est un modèle **imbriqué** ; la dernière couche
  Conv2D est cherchée *dans* le backbone, et la branche conv + la prédiction sont produites
  par **la même passe forward** (sinon le gradient est `None`).

## Principes de conception (issus de la revue du TP)

1. **Le LLM ne décide jamais d'un label, d'un score ou d'un refus.** Il propose *quels*
   outils appeler ; le code injecte les arguments, exécute, fusionne et refuse.
2. **Le gate modal est déterministe** : sans image, le CNN n'est jamais exposé ni appelé
   (exigence explicite du PDF).
3. **Fusion par règles**, jamais une moyenne de scores : concordance / contradiction /
   insuffisance, avec seuils `TAU_HIGH` / `TAU_LOW` **à calibrer sur le split validation**.
4. **Provenance** : tout nombre présent dans la réponse doit venir d'un résultat d'outil ;
   sinon on retombe sur la réponse déterministe et on journalise la violation.
5. **Journal JSONL** (`outputs/agent_log.jsonl`) : appels d'outils (arguments, durée,
   `forced`, `fallback`), décisions de fusion, trace complète par tour.
6. **Avertissement non diagnostique** ajouté par le code, pas par le LLM.

## Mise en route

```powershell
# 1) dépendances
python -m pip install mcp pytest

# 2) modèle local (déjà installé sur cette machine)
ollama list            # doit afficher qwen2.5:3b
# ollama pull qwen2.5:3b

# 3) poids CNN (Keras 3 : config.json + metadata.json + model.weights.h5 dans un .keras)
#    soit via l'API Kaggle (kernel abdelraoufgheyat/kermany-nih-merged) :
python t3_agentic/scripts/fetch_assets.py
#    soit manuellement : onglet Output du kernel -> telecharger densenet_merged.keras
#    -> decompresser dans weights/ (le dossier doit contenir config.json etc.)
#    -> ou zipper le dossier en weights/densenet_merged.keras (ce que fait load_model)

# 4) corpus d'images de test (images BRUTES : le serveur applique le pretraitement)
python t3_agentic/scripts/build_demo_corpus.py
# -> demo_images/kermany_test/{NORMAL,PNEUMONIA} (test officiel Kermany, jamais vu a l'entrainement)
# -> demo_images/nih_eval/{Consolidation,Infiltration,Pneumonia,No_Finding} (echantillon NIH, domain shift)

# 5) tests
python -m pytest t3_agentic/tests -q

# 6) un tour d'agent
python -m t3_agentic.agent.agent "Que peut-on dire de cette radio ?" `
  --image demo_images/kermany_test/PNEUMONIA/PNEUMONIA_0.jpeg

# 7) evaluation agent (22 cas : image, note, les deux, contradiction, injection, questions de connaissances)
python -m t3_agentic.eval.run_eval --out agent_eval_llm.json

# 8) ablation B4 : le meme jeu avec le routeur deterministe (sans LLM)
python -m t3_agentic.eval.run_eval --no-llm --out agent_eval_router.json
python t3_agentic/eval/ablation.py

# 9) evaluation NLP (gold sets dev / test)
python -m t3_agentic.eval.eval_nlp.py

# 10) calibration + ajustement des seuils (demi-validation, seed 42)
python t3_agentic/scripts/dump_preds.py --dir demo_images/kermany_test `
  --out outputs/preds_kermany_full.npz
python t3_agentic/eval/fit_thresholds.py      # -> outputs/calibration_config.json
                                             #    (temperature + seuil de decision + TAU_LOW/TAU_HIGH,
                                             #     ajustes sur la demi-validation Kermany, seed 42)
python t3_agentic/eval/calibration.py outputs/preds_kermany_full.npz

# 11) export du rapport Word
python t3_agentic/scripts/md_to_docx.py
```

**Pipeline complet en une commande :**

```powershell
powershell -ExecutionPolicy Bypass -File t3_agentic\run_all.ps1
# option : -SkipInference pour reutiliser les probabilites deja calculees
```

> Note sur les poids : l'API de sortie de kernel Kaggle peut renvoyer `429 Too Many Requests`
> (limite de débit sur `ListKernelSessionOutput`). Le contournement est de télécharger
> `densenet_merged.keras` depuis l'onglet **Output** du kernel dans le navigateur
> (c'est un bundle Keras 3 : `config.json` + `metadata.json` + `model.weights.h5`).

## Points de conception notables

- **Gate modal déterministe** : `allowed_tools(image, note)` — sans image, le CNN n'est
  jamais exposé ; sans image *et* sans note, l'agent lit la ressource MCP
  `guidelines://pneumonia` (outil `get_guideline`).
- **Boucle observation → décision** : après chaque appel d'outil, le résultat est renvoyé
  au LLM qui peut demander un autre outil (max 3 tours, 5 appels). Le harnais complète
  ensuite les outils obligatoires.
- **Seuils ajustés, pas choisis à la main** : `outputs/calibration_config.json` (ajustés
  sur la moitié validation du test officiel Kermany, seed 42) est lu par le serveur
  vision (température) et par la fusion (`TAU_LOW` / `TAU_HIGH`). NIH n'est **jamais**
  utilisé pour l'ajustement.
- **Provenance** : tout chiffre de la réponse doit venir d'un résultat d'outil, sinon la
  réponse déterministe est renvoyée et la violation est journalisée.
- **Écart documenté** : scispaCy n'est pas installable sur cette machine (voir
  `requirements.txt`) ; le NER est fait par règles et **mesuré** (F1 0.965 sur le split
  de test non ajusté).

> Note sur les poids : l'API de sortie de kernel Kaggle peut renvoyer `429 Too Many Requests`
> (limite de débit sur `ListKernelSessionOutput`). Le contournement est de télécharger
> `densenet_merged.keras` depuis l'onglet **Output** du kernel dans le navigateur.

## Résultat de référence (Kaggle, kernel `kermany-nih-merged`)

| Domaine | n | accuracy | AUC | rappel NORMAL | rappel PNEUMONIA |
|---|---:|---:|---:|---:|---:|
| Kermany (test, split patient) | 940 | 0.9660 | 0.9893 | 0.923 | 0.984 |
| NIH (test retenu) | 632 | 0.7199 | 0.7436 | 0.933 | 0.253 |

Rappel important : sur NIH, la classe majoritaire (normal) atteint déjà 0.687 d'accuracy ;
l'accuracy seule est donc un indicateur trompeur, l'**accuracy équilibrée** vaut ~0.593.
Le seuil de décision est désormais **ajusté** sur la demi-validation Kermany
(seed 42) : `LABEL_THRESHOLD = 0.855` sur le score calibré (≈ 0.928 sur la probabilité brute),
choisi pour maximiser l'accuracy équilibrée. Sans lui, le modèle déclarait
"pneumonie" à 73.6 % du test officiel (prévalence 62.5 %). Voir `agent/calibration.py`.

## Limites connues (à reprendre telles quelles dans le rapport)

- NIH réduit à binaire « pneumonie-like (`Pneumonia|Consolidation|Infiltration`) vs `No Finding` » ;
  l'infiltration n'est pas une pneumonie et les labels NIH sont extraits par NLP.
- Décalage de domaine confondant population (pédiatrique vs adulte) et acquisition (site/appareil).
- Le split Kermany utilisé ici est le nôtre (940 images), pas le split officiel (624) :
  ne pas comparer avec des chiffres publiés.
- Aucune validation clinique ; le système est un prototype de TP.
