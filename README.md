# Thème 3 — Agentic system for medical image analysis

Master's course (MLCI) — Thème 3: *Système agentique d'aide à l'analyse d'images médicales*.

An LLM-driven agent (local Ollama, `qwen2.5:3b`) selects tools exposed over **MCP**, fuses
their outputs with explicit rules, **refuses to conclude** when the modalities disagree or the
evidence is not trusted, journals every tool call, and is evaluated on dedicated held-out suites.

> Educational prototype. Not a medical diagnosis.

## What is here

| Path | Content |
|---|---|
| **`docs/GETTING_STARTED.md`** | **START HERE** - 10-minute setup, what each file does, how to reproduce every number, troubleshooting |
| `t3_agentic/` | the system: MCP servers (vision / NLP / resources), agent loop, evaluation harness, Streamlit UI |
| `weights/densenet_merged.keras` | **the trained CNN (34 MB, versioned)** - a clone runs the vision part without any Kaggle access |
| `docs/Report_T3_Agentic_System.md` / `.docx` | the report (English) with all measured numbers |
| `docs/superpowers/specs/` | design spec |
| `docs/superpowers/plans/` | implementation plan |
| `kermany_nih_merged.ipynb` | the notebook that trained the vision model (Kaggle kernel `abdelraoufgheyat/kermany-nih-merged`) |
| `notebookdc29842e8f_fixed.ipynb` | Kermany-only lineage notebook with the Grad-CAM fix |
| `outputs/` | every measurement as JSON (agent evals, NLP gold sets, calibration, bootstrap CIs, NIH metrics, robustness, log mining) |

## Architecture

```
user question
     |
[ deterministic modality gate ]  allowed_tools(image, note)
     |                             image / note / both / neither -> MCP resource
[ LLM proposes which tools ]      bounded to the allowed set
     |
[MCP: vision]                      [MCP: NLP]            [MCP: resources]
 classify_image / explain_image     analyze_text           guidelines://pneumonia
 (DenseNet121 + Grad-CAM)           symptoms + orientation
     |                                   |
     +-----------------+-----------------+
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
| Vision, Kermany official test (624 images, never trained on) | accuracy **0.883**, AUC **0.975**, PR-AUC 0.983, ECE 0.0376 |
| Vision, Kermany own re-split (940) | accuracy 0.966, AUC 0.989 (Kaggle) |
| Vision, NIH Chest X-ray14 (394 images, unfitted domain-shift check) | balanced accuracy **0.604**, AUC 0.735, pneumonia-like recall 0.273 |
| NLP entity extraction, held-out notes | P 1.00 / R 0.95 / F1 0.977 (dev 1.00) |
| Agent, 3 suites (22 / 18 / 10 cases) | fusion accuracy **1.000**, refusals 0.20–0.36, hallucination **0.000**, injection compliance **0.000**, warning present **1.000** |
| Agent tool choice | raw LLM 0.591 / 0.611 / 0.800 → **1.000** after harness completion |
| Robustness (official test, 624 images) | noise: accuracy 0.747, 14 % decision flips · rotation 0.894 · contrast 0.853 |
| Calibration (312 test half, 1 000 bootstrap resamples) | accuracy 0.872 [0.840, 0.904] · AUC 0.975 [0.957, 0.988] |

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
- The operating point (thresholds + temperature) is fitted on a **Kermany validation half only**
  (312/312 stratified split, seed 42) and does **not** transfer to NIH.
- The runtime keeps the PDF-mandated French wording for the non-diagnostic warning and the
  structured output; this report and the interface are in English.

## Citation of the artefacts

Training: Kaggle kernel `abdelraoufgheyat/kermany-nih-merged` (DenseNet121, T4 x2).
All measurement JSONs live in `outputs/`; the report's numbers are generated from them.
