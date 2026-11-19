# Agentic Medical Image Analysis
## Project and Oral Defense Guide

**Theme:** T3 - Agentic system for medical image analysis  
**Purpose:** Explain the architecture, model, Grad-CAM, NLP, fusion, evaluation, limitations, and live demo.  
**Safety statement:** Prototype pédagogique, ne constitue pas un diagnostic médical.

## 1. The project in one minute

This project is a local educational assistant for chest X-rays. A local LLM (Ollama with `qwen2.5:3b`) proposes which tools the agent should call. MCP servers provide image classification, clinical-note analysis, and reference material. Deterministic application code validates tool calls, combines the model and note results, refuses when evidence is insufficient or contradictory, adds the required warning, and records a tool-call journal.

The central design principle is: **the LLM may route tools, but it does not make the final medical classification or fusion decision.** For image and note cases, the final wording is assembled by a deterministic template using structured tool results. The LLM may write knowledge-only answers from retrieved reference material.

This is an educational prototype, not a medical device. A score is not a diagnosis or a clinical probability.

## 2. What the assignment asks for

The project connects a CNN, text analysis, and an agent that can choose and sequence tools. Key requirements include MCP communication, structured tool outputs, choosing tools according to the provided modalities, a persistent non-diagnostic warning, refusal when evidence is weak or contradictory, and a journal of tool calls.

The project also evaluates more than the image classifier: it tests agent routing, guardrails, prompt-injection handling, clinical-note extraction, calibration, robustness, and transfer to a second dataset.

## 3. System architecture

```text
User question + optional X-ray + optional clinical note
                         |
              Deterministic modality gate
                         |
          LLM proposes allowed MCP tools
          (or a deterministic rule router)
                         |
        +----------------+----------------+
        |                |                |
   Vision MCP         NLP MCP        Resources MCP
 classify + CAM      note parsing       guidelines
        |                |                |
        +-------- tool observations -----+
                         |
            Harness verifies required tools
                         |
               Rule-based fusion
           result, confidence, refusal
                         |
     Deterministic case answer + safety warning
                         |
                    JSONL log
```

MCP means Model Context Protocol. It gives the agent a standard way to call tools hosted in separate server processes. It is the tool interface and transport; it is not itself the image model or the LLM.

### MCP servers

| Server | Main responsibility |
|---|---|
| Vision | DenseNet classification and Grad-CAM visualization |
| NLP | Extract symptoms and negation from a written clinical note |
| Resources | Provide or retrieve general reference passages and guidelines |
| NIH | Read-only checks and information related to the NIH dataset/domain |

The agent and gate are implemented under `t3_agentic/agent/`. Vision code is under `t3_agentic/vision/`; NLP code is under `t3_agentic/nlp/`; resource and NIH servers have their own subdirectories.

## 4. The agent and the LLM

The agent uses a bounded loop:

1. The modality gate determines which tools are allowed for the inputs.
2. The LLM proposes a tool or tools.
3. The harness filters out unknown or disallowed tools and executes valid choices.
4. Tool results are returned to the agent as observations for a later round.
5. If a required tool was missed, the harness runs it anyway.
6. Deterministic fusion produces the final result or refusal.

The loop is bounded so the model cannot call tools forever. The harness, not the LLM, injects the file path and clinical note into tool calls.

The gate enforces a key safety property: with a text-only question, the image-classification tools are not offered. The LLM cannot call a hidden tool merely by being prompted to do so.

### What the LLM does and does not do

The LLM can:

- Propose which permitted tools to call.
- Phrase an answer to a knowledge-only question using retrieved references.
- Run a critic pass on knowledge-only wording, when configured.

The LLM does not:

- Read image pixels or classify the X-ray.
- Extract symptoms from the image.
- Choose the final image label or calibrated score.
- Decide whether image and note contradict.
- Override the deterministic refusal or safety warning.

For image or note cases, the answer is generated from code and structured tool results, even if the tool journal says the LLM proposed the tools. Therefore, `tool_choice_source: "llm"` means the LLM routed tools; it does **not** mean it wrote the image-case answer.

To verify LLM use, inspect `outputs/agent_log.jsonl`:

| Journal value | Meaning |
|---|---|
| `llm` | The LLM proposed at least one allowed tool |
| `fallback` | The LLM call failed or proposed nothing usable; fallback routing followed |
| `rule_router` | Deterministic routing was selected |

The harness can still complete missing required tools, so raw LLM routing accuracy and final system correctness are different measures.

## 5. Image model and preprocessing

The vision model is a binary DenseNet121 classifier. Its output is one score for the pneumonia class. The model does not generate a radiology report, name a lesion, or identify symptoms.

The training and inference preprocessing must match:

1. Crop approximately 5% from the image border.
2. Resize to 256 x 256.
3. Stretch intensities using the 1st and 99th percentiles.
4. Apply CLAHE local contrast enhancement.
5. Convert grayscale to three channels and scale to the model input range.

Inference uses ONNX with DirectML when the exported ONNX model and compatible device are available; otherwise it falls back to Keras/TensorFlow. Grad-CAM requires TensorFlow gradients, so the ONNX path cannot generate the heatmap.

### Training and splits

The training work uses the Kermany pediatric chest X-ray dataset and NIH ChestX-ray14. The model is DenseNet121 initialized with ImageNet weights and fine-tuned in two phases: first the classification head, then selected backbone layers while Batch Normalization layers remain frozen. Training used a patient-level split and early stopping based on validation loss.

The rebuilt Kermany pool contains 5,824 deduplicated images: 4,013 training, 871 validation, and 940 test images (seed 42). The threshold and calibration parameters are fitted on the validation split, not the held-out 940-image test split.

**Important evaluation caveat:** the Kaggle training notebook collected the original Kermany train, validation, and test folders before re-splitting the combined pool. Therefore, the original 624-image official test folder is not the clean independent hold-out. Report results with their exact split names; do not describe that official folder result as the untouched test result.

## 6. Calibration and classification

The raw CNN output is not treated as a clinical probability. Temperature scaling adjusts calibration, and a threshold fitted on the validation data converts the calibrated score into the model's binary class.

Current documented fitted settings:

| Parameter | Value | Purpose |
|---|---:|---|
| Temperature | 0.8109 | Calibration by validation negative log-likelihood |
| Label threshold | 0.685 | Binary decision boundary |
| `tau_low` | 0.685 | Minimum trusted image score for fusion |
| `tau_high` | 0.9126 | Confidence category boundary |

For example, a calibrated pneumonia score of 71.4% is above the 68.5% label threshold, so the model predicts pneumonia. It is below the 91.26% high-confidence boundary, so the system describes confidence as moderate. These are model operating rules, not medical risk estimates.

The threshold is a trade-off: changing it changes sensitivity and specificity. A screening system might choose a different operating point, but that choice would need validation for its intended population and use.

## 7. What Grad-CAM does

Grad-CAM (Gradient-weighted Class Activation Mapping) is a gradient-based visualization of which feature-map regions influence a selected class score in this model.

In this implementation:

1. The code finds the last convolutional layer inside the nested DenseNet backbone.
2. It runs the feature map and output through one forward pass.
3. TensorFlow's gradient tape computes the selected class score's gradients with respect to that feature map.
4. The gradients are averaged spatially to get channel weights.
5. The weighted feature maps are summed, passed through ReLU, and normalized to produce a coarse map (currently 8 x 8).
6. The map is enlarged and smoothed for display over the X-ray.

The smooth rendering is only a display choice. It adds no spatial detail to the underlying coarse map.

### What Grad-CAM cannot establish

Grad-CAM explains model sensitivity, not patient physiology. It may highlight anatomy, borders, labels, devices, or dataset-specific shortcuts. It does not prove that a highlighted region is abnormal, identify symptoms, or localize pneumonia.

The thorax-mask overlap (`inside_ratio`) is a geometric statistic comparing the map with an approximate chest mask. In the current implementation it is descriptive only; it does not change the label, fusion result, or confidence category. It must not be presented as clinical evidence or as a validated reliability score.

If asked whether Grad-CAM is faithful, answer carefully: it is a post-hoc attribution heuristic for this model's score, not a validated explanation of the patient's condition.

## 8. NLP and clinical notes

The NLP component is a rule-based symptom lexicon, not a large language model. It recognizes selected French and English terms, normalizes accents, handles common negation patterns, and applies rules to determine whether the note is sufficient for a limited respiratory orientation. Explicit diagnostic mentions are scrubbed before symptom analysis.

Examples:

| Note text | Expected interpretation |
|---|---|
| `fever and cough for 3 days` | Fever and cough detected in the note |
| `pas de fièvre, pas de toux` | Fever and cough explicitly denied |
| `sans toux, avec fièvre` | Cough denied; fever reported |
| Empty note | No symptoms reported; the system cannot infer them from the X-ray |

Symptoms are extracted from the note only. The classifier cannot detect fever or cough from pixels.

The note benchmark reports entity F1 around 0.977 on annotated in-distribution notes. This does not mean the lexicon understands every free-text note; misspellings, unusual phrasing, and clinical context remain limitations.

## 9. Fusion and refusal

Fusion is a deterministic function in `t3_agentic/agent/fusion.py`. It compares the calibrated image score and the note's rule-based orientation; it does not average their scores.

| Evidence | Typical outcome |
|---|---|
| Image score below `tau_low` | Insufficient evidence; abstain |
| Trusted image and sufficient note disagree | Contradiction; abstain and request human review |
| Trusted image only | Image-only experimental suspicion, with model confidence category |
| Sufficient note only | Low-confidence text-only suspicion; not an image result |
| No usable modality | Refuse to conclude |

Grad-CAM overlap no longer affects fusion. A normal/pneumonia prediction is a model class, not a diagnosis.

## 10. Evaluation results

Always report the dataset and split beside a metric. The following values come from the current project notes:

| Evaluation slice | Accuracy | AUC | Recall normal | Recall pneumonia |
|---|---:|---:|---:|---:|
| Clean patient-disjoint Kermany test, 940 images | 0.9649 | 0.9893 | 0.9412 | 0.9746 |
| Official Kermany folder, 624 images, included in the combined training pool | 0.9054 | 0.9750 | 0.7821 | 0.9795 |
| NIH-632 held-out external slice | 0.7199 | 0.7436 | 0.9330 | 0.2530 |
| NIH-394 local sample | 0.5480 | 0.7350 | 0.9600 | 0.1240 |

The model transfers much less reliably to the adult NIH population. Its ranking AUC remains above chance, but the chosen operating point has very low pneumonia recall on the local NIH sample. This is a major domain-shift limitation, not a result to hide. Do not claim pediatric validation proves adult performance.

On the clean Kermany test, the project notes report ECE 0.0036 after calibration versus 0.0151 before calibration at the documented comparison point. ECE is a calibration summary; it does not prove clinical safety.

### Agent evaluation

The project notes report three agent suites:

| Suite | Raw LLM tool choice | After harness | Fusion correctness | Warning present |
|---|---:|---:|---:|---:|
| 100-query suite | 0.920 | 1.000 | 1.000 | 1.000 |
| Held-out, 18 cases | 0.722 | 1.000 | 1.000 | 1.000 |
| Fresh, 10 cases | 0.900 | 1.000 | 1.000 | 1.000 |

These results show the value of the deterministic gate and harness: they validate and complete tool execution even when the LLM's proposal is imperfect. The result is not evidence that an LLM independently makes medically correct decisions.

## 11. Safety, ethics, and limitations

- The system always adds the educational non-diagnostic warning.
- The gate limits tools to the modalities actually provided.
- Fusion abstains on low image confidence or trusted image-note contradiction.
- Clinical note contents are treated as data, not as instructions to the agent.
- The deterministic answer uses structured model and NLP results; the LLM is not allowed to invent the final class.
- All processing is intended to be local; the user should still avoid entering identifiable patient data in a classroom prototype.
- Kermany labels are not radiologist-verified per image, and NIH labels are noisy label-derived categories.
- There is a pediatric-to-adult domain shift.
- Grad-CAM is not lesion localization and should not be used to justify an image prediction.
- The system is not validated for patient care and must not replace a clinician.

The synthetic `reports_dataset.csv` and `reports_dataset.jsonl`, if present, contain generated wording/examples. They are not a source of real X-ray labels or clinical ground truth and should not be used to train the CNN. They are not automatically wired into the runtime answer path.

## 12. Suggested live demo

1. Start the UI and check that the LLM checkbox is enabled if demonstrating LLM routing.
2. Upload a Kermany chest X-ray and submit it with no clinical note.
3. Show the CNN's model prediction, calibrated score, decision threshold, and confidence category.
4. Explain that the Grad-CAM is a coarse model-attribution view. Point out that it is not a lesion map and that the mask overlap does not change the decision.
5. Run the same image with a note such as `pas de fièvre, pas de toux` to demonstrate NLP extraction and, when the evidence is trusted and contradictory, refusal by fusion.
6. Open the tool-call journal. `tool_choice_source: llm` means the LLM proposed tools; the image-case answer is still deterministic.

Do not use a single heatmap or one image as evidence of clinical performance. Use the held-out evaluation table for performance claims.

## 13. Likely teacher questions

**What does MCP do?**  
MCP is the protocol that lets the agent call tools hosted by separate servers. The vision server runs classification and Grad-CAM; the NLP server analyses notes; resources provide reference text.

**Does the LLM classify the X-ray?**  
No. The CNN classifies the image. The LLM proposes tools. Deterministic code validates the tools, fuses structured results, and formats image/note answers.

**How do you stop a text question from invoking the CNN?**  
The modality gate filters the available tools before the LLM sees them. The harness also checks tool choices and only runs required tools for available inputs.

**How is Grad-CAM computed?**  
Gradients of the selected class score are taken with respect to the last convolutional feature map. Spatially averaged gradients weight the channels; a ReLU weighted sum yields a coarse attribution map.

**Does the hot region show pneumonia?**  
No. It shows regions that influence this model's score. It is not a lesion detector or patient finding.

**Where do symptoms come from?**  
Only from the clinical note, using a rule-based lexicon with negation handling. The image model does not identify symptoms.

**Why can the system refuse?**  
To avoid presenting a conclusion when image confidence is below the configured trust threshold or when trusted image and note evidence contradict.

**Why are NIH results weaker?**  
NIH is a different population and image distribution (adult rather than pediatric Kermany), with noisy labels. The Kermany operating threshold does not transfer well.

**How do you know the LLM was used?**  
Inspect `tool_choice_source` in `outputs/agent_log.jsonl`. `llm` means it proposed allowed tools; it does not prove that the LLM wrote a case-specific answer.

## 14. One-minute defense summary

> We built a local agentic chest X-ray prototype. A deterministic modality gate limits which MCP tools are available, while a local LLM proposes tool calls. DenseNet121 classifies the image, a rule-based NLP component extracts symptoms from a written note, and deterministic fusion produces a cautious result or refuses when evidence is weak or contradictory. Grad-CAM is shown only as a coarse model-attribution visualization, never as lesion localization. We evaluate on patient-disjoint Kermany data and report the weaker NIH transfer result as an explicit domain-shift limitation. The system is educational and not a medical diagnosis.

## 15. Useful commands

Start the UI on Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
python -m streamlit run t3_agentic/ui/app.py --server.port 8001
```

Run tests:

```powershell
python -m pytest t3_agentic/tests -q
```

Check LLM tool-routing evidence:

```powershell
Select-String -Path outputs/agent_log.jsonl -Pattern '"tool_choice_source": "llm"'
```

Use `tool_choice_source`, dataset split, and metric definitions precisely when presenting results. Never describe a model score or Grad-CAM map as a clinical diagnosis.
