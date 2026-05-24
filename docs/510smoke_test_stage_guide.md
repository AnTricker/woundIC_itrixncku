# 510 Smoke Test Stage Guide

This stage turns the prompt pipeline into a reproducible experiment. The goal is not only to generate captions, but to record enough data to compare runs, debug Gemini API limits, and evaluate the local model against a SaaS reference.

Core idea:

```text
same image
-> SaaS VLM output = reference baseline
-> local VLM output = candidate
-> same-mode scoring
-> local model evaluation
```

## 1. Why This Stage Exists

The 10% smoke test is a fast check before any formal full-dataset run.

It answers:

- Can the pipeline finish on the selected sample?
- Can both `simple` and `full` prompts produce valid JSON?
- Can local and SaaS outputs be generated for the same images?
- Is the Gemini baseline stable enough to use as reference?
- Are score, runtime, quota, retry, and failure records saved?
- Does `full` prompt improve the local model score compared with `simple`?

Important rule:

```text
Official local score uses same-mode reference only.
```

Valid official scores:

| Candidate | Reference | Meaning |
|---|---|---|
| `local simple` | `saas simple` | local model quality under simple prompt |
| `local full` | `saas full` | local model quality under full prompt |

Not official local scores:

- `local simple` vs `saas full`
- `local full` vs `local simple`

Those mix model effect and prompt effect, so they do not answer the project's core evaluation question cleanly.

## 2. Prepare 10% Manifest

Command:

```bash
python -m pipeline prepare \
  --image-dir images \
  --run-dir runs/510_smoke_v1 \
  --smoke-percent 10 \
  --exclude-dirs MM-SkinQA \
  --seed 42
```

Meaning:

- `prepare` scans the dataset and writes the image manifest.
- `--image-dir images` points to the original wound images.
- `--run-dir runs/510_smoke_v1` is the experiment folder.
- `--smoke-percent 10` marks about 10% of supported images as `smoke`.
- `--exclude-dirs MM-SkinQA` keeps MM-SkinQA out of this visual evaluation set.
- `--seed 42` makes the smoke sample reproducible.

Outputs:

```text
runs/510_smoke_v1/manifest.jsonl
runs/510_smoke_v1/excluded_manifest.jsonl
runs/510_smoke_v1/split_summary.json
```

Use `split_summary.json` to confirm sample count and category distribution before generation.

## 3. Run Local Smoke First

Command:

```bash
python -m pipeline generate \
  --run-dir runs/510_smoke_v1 \
  --local-scopes smoke \
  --saas-scopes none \
  --modes simple,full \
  --local-model gemma4:e2b \
  --workers 1
```

Meaning:

- `--local-scopes smoke` runs local generation on the smoke subset.
- `--saas-scopes none` skips SaaS calls completely.
- `--modes simple,full` generates both prompt variants.
- `--workers 1` is the safest starting point for local GPU / Ollama stability.

Outputs:

```text
runs/510_smoke_v1/outputs/local/simple/*.json
runs/510_smoke_v1/outputs/local/full/*.json
runs/510_smoke_v1/experiment_record.json
```

Why local first:

- It checks prompts and schema without spending API quota.
- It separates local model problems from Gemini API problems.
- It confirms the output folders and manifest matching before SaaS generation.

## 4. Run Gemini One-Image Probe

Before running Gemini on the whole smoke subset, test one randomly selected smoke image with both `simple` and `full`.

Command:

```bash
python -m pipeline gemini-probe \
  --run-dir runs/510_smoke_v1 \
  --scopes smoke \
  --seed 42 \
  --saas-model gemini-2.5-flash \
  --saas-request-delay-sec 2 \
  --saas-max-retries 5 \
  --saas-backoff-base-sec 3
```

Meaning:

- `gemini-probe` randomly selects one image from the selected scope.
- It runs Gemini `simple` once on that image.
- It runs Gemini `full` once on the same image.
- `simple` is expected to use 1 Gemini request per image.
- `full` is expected to use 3 Gemini requests per image: caption, VQA, self-check.
- It records timing, request count, retry count, rate-limit/quota errors, schema validity, quota-consuming requests, and Gemini `usageMetadata` token counts.
- It projects the approximate smoke-subset consumption for `simple`, `full`, and `simple+full`, including configured sleep time and model RPM/RPD/TPM limits.

Outputs:

```text
runs/510_smoke_v1/probes/<probe_id>/simple.json
runs/510_smoke_v1/probes/<probe_id>/full.json
runs/510_smoke_v1/probes/<probe_id>/gemini_probe_record.json
runs/510_smoke_v1/experiment_record.json
```

Important fields in `gemini_probe_record.json`:

- `selection`: which image was selected and how many images are in the target scope.
- `results.simple.probe`: one-image simple call metadata.
- `results.full.probe`: one-image full call metadata.
- `results.*.probe.quota_consuming`: request quota and TPM-related usage.
- `results.*.probe.token_count`: Gemini API token count from `usageMetadata`.
- `prediction.modes.simple`: projected simple smoke usage.
- `prediction.modes.full`: projected full smoke usage.
- `prediction.combined_simple_full`: projected usage if both modes are run.
- `prediction.model_limits`: RPM/RPD/TPM limits used for the prediction.

Use this step to answer:

- Can Gemini API key and model call work today?
- Does simple/full output pass schema on at least one image?
- How many requests will the smoke subset roughly need?
- Will the projected request count exceed RPD?
- Does the configured request delay stay under the model RPM limit?
- Is projected input token/min under TPM?
- How long may the Gemini smoke generation take?
- Is the current delay/backoff enough, or do rate-limit errors appear immediately?

If the probe fails, do not run the full Gemini smoke subset yet. First adjust delay, retry, quota, model name, or API key.

Probe outputs are only a preflight check. They do not replace the official SaaS smoke outputs under `outputs/saas/simple` and `outputs/saas/full`, and they are not used as scoring references.

## 5. Run Gemini Simple

Command:

```bash
python -m pipeline generate \
  --run-dir runs/510_smoke_v1 \
  --local-scopes none \
  --saas-scopes smoke \
  --modes simple \
  --saas-provider gemini \
  --saas-model gemini-2.5-flash \
  --workers 1 \
  --saas-request-delay-sec 2 \
  --saas-max-retries 5 \
  --saas-backoff-base-sec 3
```

Meaning:

- `--local-scopes none` skips local generation.
- `--saas-scopes smoke` runs Gemini only for smoke images.
- `--modes simple` starts with the cheaper one-call prompt.
- `--saas-request-delay-sec 2` waits before SaaS requests.
- `--saas-max-retries 5` retries temporary rate-limit or network failures.
- `--saas-backoff-base-sec 3` increases retry wait time exponentially.

Output:

```text
runs/510_smoke_v1/outputs/saas/simple/*.json
```

Use this step to detect whether the limit problem is request frequency, quota, or prompt cost.

## 6. Run Gemini Full

Command:

```bash
python -m pipeline generate \
  --run-dir runs/510_smoke_v1 \
  --local-scopes none \
  --saas-scopes smoke \
  --modes full \
  --saas-provider gemini \
  --saas-model gemini-2.5-flash \
  --workers 1 \
  --saas-request-delay-sec 2 \
  --saas-max-retries 5 \
  --saas-backoff-base-sec 3
```

Meaning:

- `full` uses multiple stages, so it costs more requests per image.
- Run it after `simple` so API errors are easier to locate.
- Keep `workers 1` until Gemini is stable.

Output:

```text
runs/510_smoke_v1/outputs/saas/full/*.json
```

## 7. Score Smoke Outputs

Command:

```bash
python -m pipeline score \
  --run-dir runs/510_smoke_v1 \
  --reference-mode full \
  --score-scopes smoke
```

Meaning:

- `score` does not call any model.
- It reads existing local and SaaS JSON outputs.
- It calculates official same-mode local scores:
  - `local_simple_vs_saas_simple`
  - `local_full_vs_saas_full`
- `--reference-mode full` selects which SaaS mode is checked by the baseline quality gate.

Outputs:

```text
runs/510_smoke_v1/scores.json
runs/510_smoke_v1/scores.csv
runs/510_smoke_v1/scores.md
runs/510_smoke_v1/experiment_record.json
```

Read `scores.md` first for a human-friendly table. Use `scores.json` for detailed same-mode comparison records.

## 8. Generate Charts

Command:

```bash
python -m pipeline charts \
  --run-dir runs/510_smoke_v1
```

Meaning:

- Reads `scores.json` and `experiment_record.json`.
- Writes quick visual checks for report and debugging.
- Uses `matplotlib` when available, with a built-in PNG fallback when it is not installed.

Outputs:

```text
runs/510_smoke_v1/charts/json_valid_rate.png
runs/510_smoke_v1/charts/score_comparison_table.png
runs/510_smoke_v1/charts/category_counts.png
runs/510_smoke_v1/charts/matched_image_count.png
runs/510_smoke_v1/charts/saas_api_errors.png
runs/510_smoke_v1/charts/same_mode_local_score_delta.png
runs/510_smoke_v1/charts/score_by_category.png
runs/510_smoke_v1/charts/latency_by_provider_mode.png
runs/510_smoke_v1/charts/tokens_per_second_local.png
```

## 9. Inspect Experiment Record

Open:

```text
runs/510_smoke_v1/experiment_record.json
```

Important sections:

- `run_metadata`: dataset, smoke percent, seed, sample count, category counts.
- `generation`: provider/model, success/failure/retry counts, Gemini API error counts.
- `gemini_probe`: one-image simple/full Gemini probe plus consumption prediction.
- `score_summary`: local simple/full score summaries.
- `evaluation_comparisons`: official same-mode comparisons and prompt-effect delta.
- `baseline_quality_gate`: whether SaaS reference is approved for scoring.
- `charts`: chart output list after running `charts`.

Baseline gate rule:

```text
approved_for_scoring = false until manual review is passed
```

Even if schema and field-completion metrics pass, the record stays exploratory until the SaaS baseline is manually reviewed.

## 10. Gemini Limit Debugging

If Gemini hits limit errors:

1. Keep `--workers 1`.
2. Increase `--saas-request-delay-sec`, for example from `2` to `5`.
3. Increase `--saas-backoff-base-sec`, for example from `3` to `10`.
4. Run `--modes simple` and `--modes full` separately.
5. Run a smaller smoke sample with a new run dir if needed.

Useful indicators in `experiment_record.json`:

- `request_count`
- `api_success_count`
- `api_failure_count`
- `success_count`
- `retry_count`
- `rate_limit_errors`
- `quota_errors`
- `network_errors`
- `failure_count`

Interpretation:

- `request_count`: API/model call attempts. Gemini `simple` is usually 1 request per image; Gemini `full` is usually 3 requests per image.
- `api_success_count`: API/model calls that succeeded.
- `api_failure_count`: API/model calls that still failed after retry.
- `success_count`: completed output jobs/files. For example, 1 Gemini `full` image can have `request_count = 3` but `success_count = 1`.
- `failure_count`: output jobs/files that did not complete.
- Many `rate_limit_errors`: requests are too close together.
- Many `quota_errors`: daily/project quota may be exhausted.
- Many retries but eventual success: delay/backoff is working.
- Failures after retries: reduce scope or wait for quota reset.

## 11. Optional One-Command Run

After the staged flow is stable, this command can run prepare, local/SaaS generation, scoring, and charts:

```bash
python -m pipeline run-all \
  --image-dir images \
  --run-dir runs/510_smoke_v1 \
  --smoke-percent 10 \
  --local-scopes smoke \
  --saas-scopes smoke \
  --modes simple,full \
  --local-model gemma4:e2b \
  --saas-provider gemini \
  --saas-model gemini-2.5-flash \
  --workers 1 \
  --saas-request-delay-sec 2 \
  --saas-max-retries 5 \
  --saas-backoff-base-sec 3
```

For first-time execution, the staged commands are safer because they isolate local prompt problems from Gemini quota problems.

## 12. Output Meaning

| Path | Meaning |
|---|---|
| `manifest.jsonl` | One row per supported image, with `smoke` / `formal` scopes |
| `split_summary.json` | Dataset counts, smoke count, category distribution |
| `outputs/local/simple` | Local candidate outputs for simple prompt |
| `outputs/local/full` | Local candidate outputs for full prompt |
| `probes/<probe_id>/simple.json` | One-image Gemini simple probe output |
| `probes/<probe_id>/full.json` | One-image Gemini full probe output |
| `probes/<probe_id>/gemini_probe_record.json` | Probe metadata and smoke consumption prediction |
| `outputs/saas/simple` | SaaS reference outputs for simple prompt |
| `outputs/saas/full` | SaaS reference outputs for full prompt |
| `scores.json` | Machine-readable score and comparison record |
| `scores.csv` | Spreadsheet-friendly score table |
| `scores.md` | Human-readable score summary |
| `experiment_record.json` | Main experiment record for reproducibility |
| `charts/*.png` | Visual analysis outputs |

## 13. Minimum Completion Criteria

This stage is complete when:

- 10% smoke manifest exists.
- Local simple/full outputs exist.
- Gemini one-image probe exists, or a probe blocker is clearly recorded.
- Gemini simple/full outputs exist or a Gemini blocker is clearly recorded.
- `scores.json`, `scores.csv`, and `scores.md` exist.
- `experiment_record.json` includes same-mode comparison records.
- Charts are generated.
- SaaS baseline quality gate is explicitly `pending`, `pass`, or `fail`.
- The report states whether `local_full_vs_saas_full` improves over `local_simple_vs_saas_simple`.
