# Thème 3 — Agentic system for medical image analysis

Master's course (MLCI) — Thème 3: *Système agentique d'aide à l'analyse d'images médicales*.

An LLM-driven agent (local Ollama, `qwen2.5:3b`) selects tools exposed over **MCP**, fuses
their outputs with explicit rules, **refuses to conclude** when the modalities disagree or the
evidence is not trusted, journals every tool call, and is evaluated on dedicated held-out suites.
The same repo carries the second-dataset (NIH) read-only MCP server for the trinôme.

> Educational prototype. Not a medical diagnosis.

## What is here

| Path | Content |
|---|---|
| **`docs/GETTING_STARTED.md`** | **START HERE** - 10-minute setup, what each file does, how to reproduce every number, troubleshooting |
| `t3_agentic/` | the system: MCP servers (vision / NLP / resources / NIH), agent loop, evaluation harness, Streamlit UI |
| `weights/densenet_merged.keras` | **the trained CNN (34 MB, versioned)** - a clone runs the vision part without any Kaggle access |
| `docs/Report_T3_Agentic_System.md` / `.docx` | the report (English) with all measured numbers, four defects documented and fixed |
| `docs/superpowers/specs/` | design spec |
| `docs/superpowers/plans/` | implementation plan |
| `kermany_nih_merged.ipynb` | the notebook that trained the vision model (Kaggle kernel `abdelraoufgheyat/kermany-nih-merged`) |
| `notebookdc29842e8f_fixed.ipynb` | Kermany-only lineage notebook with the Grad-CAM fix |
| `outputs/` | every measurement as JSON (agent evals — including the 100-query suite — NLP gold sets, calibration, bootstrap CIs, NIH metrics, robustness, log mining) |

## Architecture

```
user question
     |
[ deterministic modality gate ]  allowed_tools(image, note)
     |                             image / note / both / neither -> MCP resource
[ LLM proposes which tools ]      bounded to the allowed set
     |
[MCP: vision]                      [MCP: NLP]            [MCP: resources]         [MCP: nih]
 classify_image / explain_image     analyze_text           get_guideline            domain-shift report
 (DenseNet121 + Grad-CAM)           symptoms + orientation  search_guidelines (TFIDF),nih_findings / nih_case_lookup
                                                                    guidelines read in every analysed case
     |                                   |                          |                    |
     +-----------------+-----------------+--------------------------+--------------------+
                       |
        [ OBSERVATION -> next decision ]  (max 3 rounds)
                       |
        [ harness completion ]  (required tools that were not called)
                       |
        [ rule-based fusion ] -> {coherence, final, reason, confidence, refused}
                       |
        [ answer + provenance check + non-diagnostic warning ] + JSONL journal
```

Key design rules:

1. **The LLM never decides a label, a score or a refusal.** It proposes tools and it rephrases
   the fusion output; the code injects arguments, executes, fuses and refuses.
2. **The modality gate is deterministic** — a text-only question never triggers the CNN.
3. **Fusion is rule-based**, never an average of incomparable scores.
4. **Every number in the answer must come from a tool result** (provenance check).
5. **The journal records which path fired** (`tool_choice_source`: `llm` vs `rule_router`), so a
   silent LLM outage cannot masquerade as an ablation result.

## Headline results

| Component | Result |
|---|---|
| Vision, Kermany **clean patient-disjoint test slice** (940, rebuilt locally and verified against the Kaggle run log) | accuracy **0.965**, AUC **0.989**, balanced accuracy **0.958**, NORMAL recall 0.941 / PNEUMONIA recall 0.975, predicted-positive rate 0.710 vs prevalence 0.711 |
| Vision, Kermany official test folder (624, harder slice, inside the training pool of the deployed weights) | accuracy 0.905 at the fitted operating point (0.883 at the un-fitted 0.5), AUC 0.975, NORMAL recall 0.782 |
| Vision, NIH Chest X-ray14 (394 images, unfitted domain-shift check) | balanced accuracy **0.604**, AUC 0.735, pneumonia-like recall 0.273 |
| NLP entity extraction, held-out notes | P 1.00 / R 0.95 / F1 0.977 (dev 1.00) |
| Agent, 3 suites (22 / 18 / 10 cases) | fusion accuracy **1.000**, refusals 0.10–0.36, hallucination **0.000**, injection compliance **0.000**, warning present **1.000** |
| **Agent, 100-query suite** (50 new adversarial/accented/insufficient cases) | fusion accuracy **1.000**, tool choice raw 0.800 → **1.000** after harness completion, injection 0.000, hallucination 0.000, warning 1.000 (`outputs/agent_eval_100.json`) |
| Agent tool choice | raw LLM 0.818 / 0.722 / 0.900 → **1.000** after harness completion |
| Robustness (clean patient-disjoint test slice, 940) | noise: accuracy 0.946 · rotation 0.951 · contrast 0.962 · decision flips ≤ 0.017 |
| Calibration (clean test slice, 940; 1 000 bootstrap resamples) | accuracy 0.965 [0.952, 0.975] · AUC 0.989 [0.979, 0.996] · answered 97.9 % with 97.5 % accuracy · ECE 0.0036 |
| MC-Dropout (940 clean test images, T=10) | single pass acc 0.877 / MC-mean acc 0.878, AUC 0.974, ECE 0.036 |

## Reproduce

The trained CNN is included (`weights/densenet_merged.keras`), so a clone runs the vision
part immediately. Only the demo image corpus has to be rebuilt (optional - the measurements
are already in `outputs/`).

```powershell
python -m pip install -r t3_agentic/requirements.txt
ollama pull qwen2.5:3b
python t3_agentic/scripts/fetch_assets.py        # CNN weights (Kaggle API; manual fallback in README)
python t3_agentic/scripts/build_demo_corpus.py    # Kermany test images
python -m pytest t3_agentic/tests -q
powershell -ExecutionPolicy Bypass -File t3_agentic/run_all.ps1
python -m streamlit run t3_agentic/ui/app.py --server.port 8001
```

`fetch_assets.py` needs a Kaggle API token (`~/.kaggle/kaggle.json`); the demo corpus is public and
can also be downloaded directly from the `paultimothymooney/chest-xray-pneumonia` dataset.

## Declared deviations

- **scispaCy cannot be installed** on Python 3.13 here (it pins `spacy<3.8` and then builds
  spacy/thinc/blis from source). The entity extractor is rule/regex based and *measured*
  (F1 0.977 on a held-out note set) rather than aspirational.
- **NIH is used as binary** (Pneumonia|Consolidation|Infiltration vs No Finding) instead of the
  14-pathology multi-label setup; the NIH sample contains only 14 strictly-"Pneumonia" images.
- **Population and acquisition shift are confounded** (paediatric vs adult, different sites and
  devices): the drop to 0.604 balanced accuracy on adults cannot be attributed to one cause.
- The operating point (temperature, **decision threshold**, trust thresholds) is fitted on a
  **clean Kermany validation slice only** (871 images, patient-disjoint, seed 42, rebuilt
  locally from the notebook's own split and verified against the Kaggle run log) and does
  **not** transfer to NIH. The decision threshold is not 0.5: it is fitted for balanced
  accuracy (0.685 on the calibrated score), and the trust threshold mirrors it, so the
  system asserts a label only when it is confident enough.
- **Two Kermany slices exist and must not be mixed.** The training notebook's pool
  contains Kermany's official `test/` folder (verified: the Kaggle kernel source is
  identical to the notebook in this repo), so the official folder is not a clean hold-out
  for the deployed weights; the notebook's own patient-disjoint split (940) is, and it is
  reproduced locally by `python t3_agentic/eval/reproduce_kaggle_split.py`. The model does
  not memorise its training set (`eval/memorisation_check.py`: 0.9651 on the train slice
  vs 0.9660 on the unseen slice), so the official-folder numbers are not inflated — they
  are simply a harder population.
- The runtime keeps the PDF-mandated French wording for the non-diagnostic warning and the
  structured output; this report and the interface are in English.

## Citation of the artefacts

Training: Kaggle kernel `abdelraoufgheyat/kermany-nih-merged` (DenseNet121, T4 x2).
All measurement JSONs live in `outputs/`; the report's numbers are generated from them.
