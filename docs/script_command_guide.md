# Script Command Guide

This is the central command guide for the project. The stage scripts under `scripts/` are the preferred entrypoints. `pipeline.py` remains the shared implementation core and legacy CLI. `pipeline_original_legacy.py` is a saved snapshot of the original tracked pipeline for future comparison only.

Core rule:

```text
All run-specific results must stay inside the selected --run-dir.
```

New run naming rule:

```text
runs/<YYYYMMDD>_<model>
```

Example:

```text
runs/20260525_gemma4_26b
```

## Menu

| Section | Command / Script | Purpose |
|---|---|---|
| 1 | Standard form | How future script docs should be written |
| 2 | `scripts/prepare.py` | Build manifest and split summary |
| 3 | `scripts/generate_local.py` | Generate local model outputs |
| 4 | `scripts/gemini_probe.py` | Test Gemini on one image before full SaaS generation |
| 5 | `scripts/generate_saas.py` | Generate Gemini/SaaS outputs |
| 6 | `scripts/score.py` | Score local outputs against standard SaaS simple baseline |
| 7 | `scripts/charts.py` | Generate run-local charts |
| 8 | `scripts/run_all.py` | Optional one-command pipeline run |
| 9 | `scripts/metadata.py` | Build run-local metadata report |
| 10 | `scripts/translate.py` | Translate all output captions with local Ollama model |
| 11 | `scripts/compare_visual_summary_translations.py` | Compare translated visual summaries across 510 local, 523 local, and SaaS simple |
| 12 | `scripts/fix_charts.py` | Rewrite chart PNGs with clearer labels |
| 13 | `docs/score_calculation_guide.md` | Explain current BLEU-4, ROUGE-L, METEOR-lite, CIDEr-lite calculations |
| 14 | Validation | Static command checks |
| 15 | Update Record | Record when scripts are added or changed |

## 1. Standard Form

Use this shape when adding a new command or script:

```text
Purpose:
- What it does.
- Whether it calls local model, SaaS API, or no model.
- Whether it modifies original outputs.

Command:
<copy-paste command>

Inputs:
<run-local input paths>

Outputs:
<run-local output paths>

Meaning:
- What the output means.
- What the output must not be used for.
```

Documentation rules:

- Keep commands short.
- Avoid optional flags unless they solve a real repeated workflow.
- Put run-specific outputs under `runs/<run_id>/`.
- Do not write run-specific reports to `docs/`.
- If a command calls Gemini or Ollama, say so clearly.

## 2. Prepare Manifest

Purpose:

- Scans the image dataset.
- Creates the run directory.
- Writes manifest and split summary.
- Does not call any model.

Command:

```bash
python scripts/prepare.py \
  --image-dir images \
  --run-dir runs/20260525_gemma4_26b \
  --smoke-percent 10 \
  --exclude-dirs MM-SkinQA \
  --seed 42
```

Outputs:

```text
runs/20260525_gemma4_26b/manifest.jsonl
runs/20260525_gemma4_26b/excluded_manifest.jsonl
runs/20260525_gemma4_26b/split_summary.json
```

## 3. Generate Local Outputs

Purpose:

- Calls the local Ollama model.
- Generates local candidate output JSON files.
- Records local input/output token, prefill/decode time, VRAM, and GPU runtime metadata in `experiment_record.json`.

Command:

```bash
python scripts/generate_local.py \
  --run-dir runs/20260525_gemma4_26b \
  --scopes smoke \
  --modes simple,full \
  --local-model gemma4:26b \
  --workers 1
```

Outputs:

```text
runs/20260525_gemma4_26b/outputs/local/simple/*.json
runs/20260525_gemma4_26b/outputs/local/full/*.json
runs/20260525_gemma4_26b/experiment_record.json
```

## 4. Gemini Probe

Purpose:

- Calls Gemini on one selected image.
- Tests simple and full prompt once before larger SaaS generation.
- Estimates request/token/quota consumption.

Command:

```bash
python scripts/gemini_probe.py \
  --run-dir runs/20260525_gemma4_26b \
  --scopes smoke \
  --seed 42 \
  --saas-model gemini-3.1-flash-lite \
  --saas-request-delay-sec 4 \
  --saas-max-retries 5 \
  --saas-backoff-base-sec 5
```

Outputs:

```text
runs/20260525_gemma4_26b/probes/<probe_id>/simple.json
runs/20260525_gemma4_26b/probes/<probe_id>/full.json
runs/20260525_gemma4_26b/probes/<probe_id>/gemini_probe_record.json
runs/20260525_gemma4_26b/experiment_record.json
```

## 5. Generate SaaS Outputs

Purpose:

- Calls Gemini/SaaS model.
- Generates SaaS reference output JSON files.
- Records request, retry, quota, token, and error metadata.
- Current local-simple scoring can use the standard baseline in `runs/saas_simple_baseline`, so future simple scoring does not require another SaaS call.

Simple command:

```bash
python scripts/generate_saas.py \
  --run-dir runs/20260525_gemma4_26b \
  --scopes smoke \
  --modes simple \
  --saas-provider gemini \
  --saas-model gemini-3.1-flash-lite \
  --workers 1 \
  --saas-request-delay-sec 4 \
  --saas-max-retries 5 \
  --saas-backoff-base-sec 5
```

Full command:

```bash
python scripts/generate_saas.py \
  --run-dir runs/20260525_gemma4_26b \
  --scopes smoke \
  --modes full \
  --saas-provider gemini \
  --saas-model gemini-3.1-flash-lite \
  --workers 1 \
  --saas-request-delay-sec 4 \
  --saas-max-retries 5 \
  --saas-backoff-base-sec 5
```

Outputs:

```text
runs/20260525_gemma4_26b/outputs/saas/simple/*.json
runs/20260525_gemma4_26b/outputs/saas/full/*.json
runs/20260525_gemma4_26b/experiment_record.json
```

## 6. Score Outputs

Purpose:

- Does not call any model.
- Reads local outputs from the current run.
- Reads the standard SaaS simple baseline from `runs/saas_simple_baseline`.
- Scores local full against run-local SaaS full only if that reference exists.

Command:

```bash
python scripts/score.py \
  --run-dir runs/20260525_gemma4_26b \
  --reference-mode simple \
  --score-scopes smoke \
  --saas-simple-baseline-dir runs/saas_simple_baseline
```

Outputs:

```text
runs/20260525_gemma4_26b/scores.json
runs/20260525_gemma4_26b/scores.md
runs/20260525_gemma4_26b/experiment_record.json
```

Meaning:

- Official simple comparison is `local simple` vs standard `saas simple`.
- Standard SaaS simple baseline path: `runs/saas_simple_baseline`.
- `local full` still requires a same-mode `saas full` reference; otherwise full score is not evaluable.
- `--reference-mode simple` chooses the baseline quality gate mode; it does not create cross-mode scoring.

## 7. Generate Charts

Purpose:

- Does not call any model.
- Reads scores and experiment record.
- Writes visual analysis files under the run.

Command:

```bash
python scripts/charts.py \
  --run-dir runs/20260525_gemma4_26b
```

Outputs:

```text
runs/20260525_gemma4_26b/charts/*.png
```

## 8. Optional Run-All

Purpose:

- Runs prepare, generation, score, and charts in one command.
- Use only after staged commands are stable.

Command:

```bash
python scripts/run_all.py \
  --image-dir images \
  --run-dir runs/20260525_gemma4_26b \
  --smoke-percent 10 \
  --local-scopes smoke \
  --saas-scopes smoke \
  --modes simple,full \
  --local-model gemma4:26b \
  --saas-provider gemini \
  --saas-model gemini-3.1-flash-lite \
  --workers 1 \
  --saas-request-delay-sec 4 \
  --saas-max-retries 5 \
  --saas-backoff-base-sec 5 \
  --reference-mode simple \
  --saas-simple-baseline-dir runs/saas_simple_baseline
```

## 9. Metadata Report

Purpose:

- Does not call any model.
- Reads existing run artifacts.
- Writes a run-local report-ready metadata table.

Command:

```bash
python scripts/metadata.py \
  --run-dir runs/20260525_gemma4_26b
```

Outputs:

```text
runs/20260525_gemma4_26b/20260525_gemma4_26b_record_summary.md
```

## 10. Caption Translation

Purpose:

- Calls one local Ollama model.
- Translates every JSON file under `runs/<run_id>/outputs/`.
- Writes mirrored translated JSON files under `runs/<run_id>/output_translation/`.
- After translation completes, writes a translation metadata summary table under `output_translation/`.
- Does not modify original `outputs/`.

Command:

```bash
python scripts/translate.py \
  --run-dir runs/20260525_gemma4_26b \
  --model gemma4:26b
```

Outputs:

```text
runs/20260525_gemma4_26b/output_translation/local/simple/*.json
runs/20260525_gemma4_26b/output_translation/local/full/*.json
runs/20260525_gemma4_26b/output_translation/saas/simple/*.json
runs/20260525_gemma4_26b/output_translation/saas/full/*.json
runs/20260525_gemma4_26b/output_translation/20260525_gemma4_26b_translation_summary.md
```

Meaning:

- `caption` is translated to Traditional Chinese.
- The translated file mirrors the original result JSON.
- Each new translated JSON records translation model token count, duration, prefill/decode time, and local runtime GPU/VRAM sampling.
- The translation summary table groups translated results by source provider, source prompt mode, source model, and translation model.
- Translation is for human review only and must not be used for scoring.

## 11. Visual Summary Translation Comparison

Purpose:

- Does not call any model.
- Randomly selects two translated simple-prompt results for each wound type.
- Compares only `caption.image_observation.visual_summary`.
- Lists each selected image in this order: 510 local simple, 523 local simple, SaaS simple.

Command:

```bash
python scripts/compare_visual_summary_translations.py
```

Outputs:

```text
runs/visual_summary_translation_comparison.md
```

Meaning:

- This report is for human review of translated visual summaries.
- Because it compares multiple run versions, the output belongs in the `runs/` root.
- It does not modify original outputs.
- It does not participate in scoring.

## 12. Chart Fix

Purpose:

- Does not call any model.
- Rewrites charts with clearer labels and missing-reference notes.

Command:

```bash
python scripts/fix_charts.py \
  --run-dir runs/20260525_gemma4_26b
```

Outputs:

```text
runs/20260525_gemma4_26b/charts/*.png
runs/20260525_gemma4_26b/charts/_backup_before_523_fix/*.png
```

## 13. Score Calculation Guide

Purpose:

- Explains how the current four text similarity scores are calculated.
- Written mainly in Traditional Chinese.
- Documents current implementation details, not official metric packages.

Path:

```text
docs/score_calculation_guide.md
```

Meaning:

- Use this document when interpreting `scores.json` and `scores.md`.
- It explains what the scores can and cannot prove.

## 14. Validation

Static validation:

```bash
python -m py_compile \
  pipeline.py \
  scripts/_pipeline_common.py \
  scripts/prepare.py \
  scripts/generate_local.py \
  scripts/gemini_probe.py \
  scripts/generate_saas.py \
  scripts/score.py \
  scripts/charts.py \
  scripts/run_all.py \
  scripts/metadata.py \
  scripts/translate.py \
  scripts/compare_visual_summary_translations.py \
  scripts/fix_charts.py
```

Legacy pipeline command list:

```bash
python -m pipeline --help
```

## 15. Update Record

| Date | Command / Script | Change |
|---|---|---|
| 2026-05-24 | `scripts/prepare.py`, `scripts/generate_local.py`, `scripts/gemini_probe.py`, `scripts/generate_saas.py`, `scripts/score.py`, `scripts/charts.py`, `scripts/run_all.py` | Split stage entrypoints into separate scripts while keeping `pipeline.py` as shared implementation core. |
| 2026-05-24 | `scripts/score.py` / `pipeline.py score` | Standardized simple scoring against `runs/saas_simple_baseline`. |
| 2026-05-24 | `scripts/metadata.py` | Added run-local metadata report command. |
| 2026-05-24 | `scripts/translate.py` | Translates all output JSON captions into mirrored `output_translation/`. |
| 2026-05-24 | `scripts/compare_visual_summary_translations.py` | Added translated visual-summary comparison report across 510 local, 523 local, and SaaS simple outputs. |
| 2026-05-24 | `scripts/fix_charts.py` | Documented chart rewrite command and run-local chart outputs. |
| 2026-05-24 | `docs/score_calculation_guide.md` | Added detailed Traditional Chinese explanation of BLEU-4, ROUGE-L, METEOR-lite, and CIDEr-lite calculation. |
| 2026-05-25 | `pipeline.py`, `run_simple_smoke.sh` | Added local Ollama token/timing recording, removed new `scores.csv` output, and standardized new run names as `<YYYYMMDD>_<model>`. |
| 2026-05-25 | `scripts/metadata.py` | Report filename now follows the selected run name and is written directly under that run directory. |
| 2026-05-25 | `scripts/translate.py` | Added per-translation local usage metadata and automatic `output_translation/<run-name>_translation_summary.md` report generation. |
