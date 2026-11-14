# T3 — pipeline complet (reproductible)
# Usage : powershell -ExecutionPolicy Bypass -File t3_agentic\run_all.ps1
param(
    [switch]$SkipInference   # ne pas refaire le dump des probabilites (lent ~15 min CPU)
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Push-Location $root
$env:PYTHONIOENCODING = "utf-8"

function Step($msg) { Write-Host "`n=== $msg ===" -ForegroundColor Cyan }

Step "0/9 dependances"
python -c "import mcp, tensorflow, sklearn, cv2, numpy; print('ok')"

Step "1/9 corpus de test (si absent)"
if (-not (Test-Path "demo_images\kermany_test\PNEUMONIA")) {
    python t3_agentic\scripts\build_demo_corpus.py
}

Step "2/9 poids CNN (si absent)"
if (-not (Test-Path "weights\densenet_merged.keras")) {
    python t3_agentic\scripts\fetch_assets.py
}

Step "3/9 tests unitaires"
python -m pytest t3_agentic\tests -q

if (-not $SkipInference) {
    Step "4/9 probabilites par image (inference seule, ~15 min CPU)"
    python t3_agentic\scripts\dump_preds.py --dir demo_images\kermany_test --out outputs\preds_kermany_full.npz
}

Step "5/9 ajustement des seuils + temperature (moitie validation, seed 42)"
python t3_agentic\eval\fit_thresholds.py

Step "6/9 calibration (ECE + courbe de fiabilite)"
python t3_agentic\eval\calibration.py outputs\preds_kermany_full.npz

Step "7/9 evaluation NLP (gold sets dev / test)"
python t3_agentic\eval\eval_nlp.py

Step "8/9 evaluation agent (avec LLM, puis routeur deterministe = ablation B4)"
python -m t3_agentic\eval\run_eval --out agent_eval_llm.json
python -m t3_agentic\eval\run_eval --no-llm --out agent_eval_router.json
python t3_agentic\eval\ablation.py
python t3_agentic\eval\compare_with_kaggle.py outputs\preds_kermany_full.npz

Step "9/9 robustesse (bruit / rotation / contraste) + ablation"
python t3_agentic\eval\robustness.py
python t3_agentic\eval\mine_log.py

Step "10/10 export du rapport Word"
python t3_agentic\scripts\md_to_docx.py

Write-Host "`nTermine. Sorties dans outputs\, rapport dans docs\Rapport_T3_SysA.docx" -ForegroundColor Green
Pop-Location
