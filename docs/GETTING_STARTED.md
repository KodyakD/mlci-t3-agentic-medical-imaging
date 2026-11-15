# Getting started (for the team)

You are reading this because you want to run or extend this project. Everything below is
verified on Windows / Python 3.13; Linux and macOS work too (paths are computed, not hardcoded).

**Start here:** the full report is `docs/Report_T3_Agentic_System.docx` (or `.md`).

---

## 1. What this project is

A **Thème 3 (MLCI)** deliverable: an *agentic system* for medical image analysis. An LLM
(local Ollama, `qwen2.5:3b`) does **not** decide anything medical. It only picks *which
tools to call* and *how to phrase* the final answer. Everything that matters — the image
label, the refusal, the confidence — is decided by deterministic code:

```
question ─► [modality gate] ─► [LLM proposes tools] ─► [execute MCP tools]
                    │                    │                        │
                    │                    └──► [harness completes the missing tools]
                    ▼                                             ▼
           image only / note only / both / resource      [rule-based fusion]
                                                                    ▼
                                            answer + "prototype, not a diagnosis" + journal
```

Four MCP servers:

| Server | Tools | What it does |
|---|---|---|
| `vision` | `classify_image`, `explain_image` | DenseNet121 (Kermany+NIH) + Grad-CAM heatmap with a thorax-plausibility flag |
| `nlp` | `analyze_text` | symptom extraction (FR/EN, **accented French included**), negation handling, clinical orientation, diagnostic-mention scrubbing |
| `resources` | `get_guideline`, `search_guidelines` | TF-IDF retrieval over a French reference corpus + `guidelines://pneumonia` for knowledge questions |
| `nih` | `nih_domain_shift`, `nih_findings`, `nih_case_lookup` | second-dataset (NIH Chest X-ray14) read-only server - the measured domain-shift report, the 14 findings and their prevalence |

---

## 2. What is in the box (and what is not)

**Included in the git repo (a clone works out of the box):**

- `weights/densenet_merged.keras` — **the trained CNN** (34 MB). No Kaggle account needed.
- `t3_agentic/` — the whole system (3 MCP servers, agent, evaluation harness, UI).
- `outputs/` — every measurement as JSON (agent evals, NLP gold sets, calibration,
  bootstrap CIs, NIH metrics, robustness, log mining) — so you can check numbers without
  re-running anything.
- `docs/Report_T3_Agentic_System.docx` — the report with all results.
- `kermany_nih_merged.ipynb` — the Kaggle notebook that trained the CNN (needs Kaggle + a GPU
  to re-run; you do **not** need to).

**Not included (rebuild if you need it):**

- `demo_images/` — the test images. Rebuild with
  `python t3_agentic/scripts/build_demo_corpus.py kermany`
  (public Kaggle dataset `paultimothymooney/chest-xray-pneumonia`; needs a free Kaggle
  account and `pip install kaggle`). The evaluation numbers in `outputs/` were produced with
  this corpus, so the *numbers* are already there without it.

---

## 3. Ten-minute quickstart

```powershell
git clone https://github.com/KodyakD/mlci-t3-agentic-medical-imaging.git
cd mlci-t3-agentic-medical-imaging

python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r t3_agentic/requirements.txt

# the LLM the agent uses (must support tool calling)
ollama pull qwen2.5:3b

# the web interface
python -m streamlit run t3_agentic/ui/app.py --server.port 8001
# -> http://127.0.0.1:8001
```

Every analyzed case also reads `guidelines://pneumonia` through MCP (the brief's step 4:
the agent reads its reference before concluding). Tool calls are journaled per call, including
the model's `tool_choice_source` (`llm` or `rule_router`).

In the UI you can: pick a demo image (or upload one), type a clinical note, press **Analyse**,
and read the answer, the fusion decision, the CNN result, the tool-call journal and the Grad-CAM
heatmap. The sidebar shows the fitted thresholds and lets you switch the LLM off (deterministic
router) for the ablation.

Without the UI:

```powershell
python -m t3_agentic.agent.agent "Que peut-on dire de cette radio ?" `
  --image demo_images\kermany_test\PNEUMONIA\PNEUMONIA_0.jpeg
```

---

## 4. Where things live

```
t3_agentic/
  vision/model.py       preprocessing (must stay identical to training) + CNN inference
  vision/gradcam.py     Grad-CAM on the NESTED DenseNet backbone (read the docstring!)
  vision/roi.py         thorax ellipse -> inside_ratio / suspect flag
  vision/server.py      MCP server "vision"
  nlp/terms.py          symptom lexicon, negation rules, orientation rule
  nlp/scrub.py          removes diagnostic mentions before analysis
  nlp/server.py         MCP server "nlp"
  resources/server.py   MCP server "resources" (guidelines, ethics, prompt, search_guidelines)
  nih/server.py         MCP server "nih" (second dataset, read-only) + corpus.py for the RAG retrieval
  agent/gate.py         modality gate + tool-choice validation + fallback router
  agent/fusion.py       the ONLY place that produces a label or a refusal
  agent/mcp_client.py   persistent MCP stdio clients (one process per server)
  agent/agent.py        the loop: gate -> LLM -> tools -> observation -> fusion -> answer
  agent/logger.py       JSONL journal
  agent/calibration.py  temperature scaling + threshold config loading
  agent/prompts.py      system prompt and tool schemas
  eval/                evaluation harness (see section 6)
  scripts/              asset builders, prediction dump, report export
  ui/app.py             Streamlit interface
  tests/                42 unit tests
outputs/                every measurement as JSON
```

---

## 5. Rules of the project (do not break these)

1. **The warning is mandatory.** Every answer ends with
   « Prototype pédagogique, ne constitue pas un diagnostic médical. » It is added in code,
   never by the LLM. Do not remove it.
2. **Fusion thresholds are fitted, not chosen.** They live in
   `outputs/calibration_config.json` and were fitted on a **Kermany validation half**
   (312/312 stratified split, seed 42). Never re-fit them on a test set.
3. **Never fit on NIH.** NIH is the *unfitted* domain-shift check. If you change the model,
   re-fit only on Kermany validation and report both domains.
4. **The modality gate is deterministic.** If you "simplify" it into a prompt instruction,
   a text-only question may start calling the CNN — the exact thing the PDF forbids.
5. **Keep `tool_choice_source` in the journal.** It is how we detected that the LLM service
   had silently been down. Without it, an evaluation can report a false ablation result.
6. **Report the split with every number — and say which one is clean.**
   The official Kermany test folder (624) was **inside the training pool** of
   `densenet_merged.keras`: the merged notebook re-splits Kermany's train+val+test
   by patient (kernel source verified identical to the repo notebook). The clean
   held-out slice of that model is the notebook's own patient-disjoint split,
   rebuilt and verified by `python t3_agentic/eval/reproduce_kaggle_split.py`.
   "official test (624)" and "own re-split" are
   different populations; do not compare them.
7. **The calibration temperature is not a patient vital.** The final-answer composer never
   receives it; new numbers going into the journal stay out of the LLM prompt as well.

---

## 6. How to reproduce each number

| Number | Command |
|---|---|
| agent evaluation (22 / 18 / 10 cases) | `python -m t3_agentic.eval.run_eval --out outputs/agent_eval_dev.json` · `--cases t3_agentic/eval/cases_heldout.json` · `--cases t3_agentic/eval/cases_fresh.json` |
| 100-query suite (trinôme) | `python -m t3_agentic.eval.run_eval --cases t3_agentic/eval/cases_100.json --out outputs/agent_eval_100.json` |
| ablation B4 (LLM vs deterministic router) | same with `--no-llm` |
| Ablation vs ground truth (image-only / text-only / multimodal) | `python t3_agentic.eval.ablation_groundtruth` |
| NLP entity P/R/F1 | `python -m t3_agentic.eval.eval_nlp` (dev / test / fresh splits) |
| threshold fitting + bootstrap CIs | `python t3_agentic.eval.fit_thresholds.py` · `python t3_agentic.eval.bootstrap_ci.py` |
| calibration curve + ECE | `python t3_agentic.eval.calibration.py outputs/preds_kermany_full.npz` |
| per-image predictions (needed for the above) | `python t3_agentic/scripts/dump_preds.py --dir demo_images/kermany_test --out outputs/preds_kermany_full.npz` |
| NIH domain-shift check | `python t3_agentic.eval.robustness.py` (robustness) · `python t3_agentic/scripts/nih_external_eval.py` (NIH) |
| journal mining / evidence index | `python t3_agentic.eval.mine_log.py` |
| the whole pipeline | `powershell -ExecutionPolicy Bypass -File t3_agentic/run_all.ps1` |

Everything the evaluation writes lands in `outputs/` as JSON; the report is generated from
those files (`python t3_agentic/scripts/md_to_docx.py`), so **the numbers are never typed by
hand**.

---

## 7. Troubleshooting

| Symptom | Cause / fix |
|---|---|
| The agent answers with the deterministic template and no LLM phrasing | Ollama is not running. Check `curl http://127.0.0.1:11434/api/tags`, then `ollama serve`. The journal shows which path fired: `tool_choice_source` = `llm` or `rule_router`. |
| `ollama pull` fails / no models | `ollama list`; the project needs `qwen2.5:3b` (it supports tool calling). |
| The UI says "image de demonstration" is empty | the corpus is missing: `python t3_agentic/scripts/build_demo_corpus.py kermany` |
| The CNN call is slow (~1.5 s/image) | expected on CPU: inference only, no GPU required. |
| `FileNotFoundError: weights/densenet_merged.keras` | the model is versioned; make sure you cloned the whole repo (it is 34 MB). |
| Port 8001 busy | `python -m streamlit run t3_agentic/ui/app.py --server.port 8010` |
| `scispacy` install fails on Python 3.13 | known and documented: it pins `spacy<3.8` and builds from source. Our NER is rule-based and measured (F1 0.977). Do not try to force it. |

---

## 8. Known limitations (read before claiming anything)

- The CNN is **strong on paediatric Kermany** (accuracy **0.965**, AUC **0.989** on the clean
  patient-disjoint test slice of 940 images, rebuilt and verified locally) and **poor on
  adults**: balanced accuracy **0.604**, pneumonia-like recall
  **0.273** on NIH. The operating point fitted on Kermany does **not** transfer.
- On NIH the missed positives sit at p ≈ 0.49 — the model is *undecided*, not confidently
  wrong, which is why the guardrail abstains instead of asserting.
- The vision model is a **binary** pneumonia classifier. It does not localise disease, does
  not distinguish bacterial from viral, and its Grad-CAM is a plausibility heuristic, not a
  guarantee.
- The NLP component is rule-based, not a trained clinical NER.
- The agent is a **prototype**: it refuses rather than guessing, but no clinical validation
  has been done.

Say these things in the defence. A project that states its limits precisely scores better than
one that hides them.
