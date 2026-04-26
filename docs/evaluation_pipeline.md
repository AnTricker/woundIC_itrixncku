# Wound VLM Evaluation Pipeline

## Goal

This evaluation pipeline turns prompt experiments into a 2x2 ablation table:

| Provider | Simple prompt | Full prompt |
| :--- | :--- | :--- |
| SaaS VLM baseline | SaaS + single-stage caption | SaaS + caption + VQA + self-check |
| Local VLM inference | Ollama + single-stage caption | Ollama + caption + VQA + self-check |

It answers two questions:

1. Does the multi-stage prompt improve a general SaaS VLM baseline?
2. Does the same prompt design improve local Ollama inference against a selected SaaS reference?

This is not a train/validate split. There is no training here. The split means:

- `baseline`: a small SaaS audit/calibration subset used to inspect whether the SaaS reference and prompt design are good enough.
- `inference`: the evaluation subset. Local and SaaS must both run on this same split if you want BLEU/ROUGE/METEOR/CIDEr.

## Four Decoupled Parts

## API Key Setup

Put API keys in the repo root `.env` file, not in shell commands.

```bash
cp .env.example .env
```

Edit `.env`:

```env
GEMINI_API_KEY=your_gemini_api_key
OPENAI_API_KEY=your_openai_api_key
```

`.env` is ignored by git. The pipeline automatically loads it from the repo root. If you want another file:

```bash
.venv/bin/python -m evaluation.pipeline --env-file /path/to/.env generate-saas ...
```

### 1. Preprocessing: split dataset

The split step scans an image directory and writes a manifest. It does not copy images.

Category inference:

- Preferred: first folder under `--image-dir`, e.g. `images/Abrasions/xxx.jpg` -> `abrasions`.
- Fallback: filename prefix for loose images directly under `--image-dir`.
- `MM-SkinQA` is excluded by default because it is not part of the wound-class evaluation set.
- Other unsupported folders are written to `excluded_manifest.jsonl` and skipped until a matching category prompt exists.

`baseline_percent + inference_percent` must equal 100. The percentages control audit cost versus evaluation coverage, not model training.

The split is exact at the whole-dataset level. For example, 10,000 images with `--baseline-percent 15` produces about 1,500 baseline images and 8,500 inference images. It does not force every category to contribute at least one baseline image, because datasets with many singleton filename-derived categories would otherwise overfill the baseline split.

```bash
.venv/bin/python -m evaluation.pipeline split \
  --image-dir images \
  --run-dir evaluation/runs/dev \
  --baseline-percent 20 \
  --inference-percent 80 \
  --exclude-dirs MM-SkinQA \
  --seed 42
```

Outputs:

- `evaluation/runs/dev/manifest.jsonl`
- `evaluation/runs/dev/split_summary.json`

### 2. Inference: local Ollama model

Run local model on the inference split, in both prompt modes.

```bash
.venv/bin/python -m evaluation.pipeline generate-local \
  --run-dir evaluation/runs/dev \
  --splits inference \
  --modes simple,full \
  --local-model qwen3.5 \
  --workers 1
```

Outputs:

- `evaluation/runs/dev/outputs/local/simple/*.json`
- `evaluation/runs/dev/outputs/local/full/*.json`

### 3. Baseline/reference generation: SaaS VLM

Run Gemini or OpenAI on both splits by default:

- `baseline`: inspect SaaS quality before trusting it.
- `inference`: create paired references for local-vs-SaaS scoring.

If you only want to audit first and avoid paying for the full eval reference, pass `--splits baseline`.

Gemini:

```bash
.venv/bin/python -m evaluation.pipeline generate-saas \
  --run-dir evaluation/runs/dev \
  --splits baseline,inference \
  --modes simple,full \
  --saas-provider gemini \
  --saas-model gemini-2.5-flash \
  --workers 1
```

OpenAI:

```bash
.venv/bin/python -m evaluation.pipeline generate-saas \
  --run-dir evaluation/runs/dev \
  --splits baseline,inference \
  --modes simple,full \
  --saas-provider openai \
  --saas-model gpt-4.1-mini \
  --workers 1
```

Outputs:

- `evaluation/runs/dev/outputs/saas/simple/*.json`
- `evaluation/runs/dev/outputs/saas/full/*.json`

Note: GPT Plus / Gemini web subscriptions are not API credentials. This pipeline uses API keys loaded from `.env`.

### 4. Score

The score step writes a 2x2 table and machine-readable metrics.

```bash
.venv/bin/python -m evaluation.pipeline score \
  --run-dir evaluation/runs/dev \
  --reference-mode full \
  --score-splits inference
```

Outputs:

- `evaluation/runs/dev/scores.json`
- `evaluation/runs/dev/scores.csv`
- `evaluation/runs/dev/scores.md`

Metrics:

- BLEU-4
- ROUGE-L
- METEOR-lite
- CIDEr-lite
- schema valid rate
- required field completion rate
- not-observed usage rate
- no-ruler size compliance rate
- safety-scope compliance rate

The `*-lite` metrics are dependency-light local approximations for quick iteration. Formal reporting should later add standard metric packages and expert review.

## Important Scoring Detail

Semantic scoring needs paired outputs for the same image.

BLEU/ROUGE/METEOR/CIDEr compare generated text against reference text. For image `burns (1).jpg`, the score can only be computed if both files exist:

- `outputs/local/<mode>/burns (1).json`
- `outputs/saas/<reference_mode>/burns (1).json`

If SaaS only runs on `baseline` and local only runs on `inference`, the pipeline can still report behavior metrics, but local-vs-SaaS BLEU/ROUGE/METEOR/CIDEr will have `paired_count = 0`.

So the recommended mechanism is:

1. Split dataset into `baseline` and `inference`.
2. Run SaaS on `baseline` first if you want cheap prompt/reference audit.
3. Once acceptable, run SaaS on `inference` too.
4. Run local on `inference`.
5. Score only `inference`, because that is the paired eval set.

If you audited only `baseline` first, create the missing inference references:

```bash
.venv/bin/python -m evaluation.pipeline generate-saas \
  --run-dir evaluation/runs/dev \
  --splits inference \
  --modes simple,full \
  --saas-provider gemini \
  --saas-model gemini-2.5-flash

.venv/bin/python -m evaluation.pipeline score \
  --run-dir evaluation/runs/dev \
  --reference-mode full \
  --score-splits inference
```

## Parallel Generation

Part 2 and part 3 can run in parallel:

```bash
.venv/bin/python -m evaluation.pipeline generate \
  --run-dir evaluation/runs/dev \
  --local-splits inference \
  --saas-splits baseline,inference \
  --modes simple,full \
  --local-model qwen3.5 \
  --saas-provider gemini \
  --saas-model gemini-2.5-flash \
  --parallel-providers \
  --workers 1
```

Use `--workers > 1` only after API rate limits and local GPU memory are stable.

## Stop Before Scoring

Use this when you want to inspect SaaS baseline quality before treating it as reference:

```bash
.venv/bin/python -m evaluation.pipeline run-all \
  --image-dir images \
  --run-dir evaluation/runs/dev \
  --baseline-percent 20 \
  --inference-percent 80 \
  --exclude-dirs MM-SkinQA \
  --seed 42 \
  --local-model qwen3.5 \
  --saas-provider gemini \
  --saas-model gemini-2.5-flash \
  --parallel-providers \
  --stop-before-score
```

With the default `--saas-splits baseline,inference`, this already creates paired inference references and then stops before score. You can inspect both:

- `outputs/saas/full/` for reference quality
- `outputs/local/full/` for local behavior

When quality is acceptable, score:

```bash
.venv/bin/python -m evaluation.pipeline score \
  --run-dir evaluation/runs/dev \
  --reference-mode full \
  --score-splits inference
```

If you intentionally ran SaaS on `baseline` only, generate SaaS references for inference images first:

```bash
.venv/bin/python -m evaluation.pipeline generate-saas \
  --run-dir evaluation/runs/dev \
  --splits inference \
  --modes simple,full \
  --saas-provider gemini \
  --saas-model gemini-2.5-flash

.venv/bin/python -m evaluation.pipeline score \
  --run-dir evaluation/runs/dev \
  --reference-mode full \
  --score-splits inference
```

## Full Pipeline Without Pause

This is useful for smoke tests or once baseline quality is trusted:

```bash
.venv/bin/python -m evaluation.pipeline run-all \
  --image-dir images \
  --run-dir evaluation/runs/dev \
  --baseline-percent 20 \
  --inference-percent 80 \
  --exclude-dirs MM-SkinQA \
  --seed 42 \
  --local-model qwen3.5 \
  --saas-provider gemini \
  --saas-model gemini-2.5-flash \
  --saas-splits baseline,inference \
  --parallel-providers \
  --reference-mode full \
  --score-splits inference
```

## Provider Notes

- `gemini` uses REST `models/{model}:generateContent` with inline image data.
- `openai` uses the Responses API with base64 image data URLs.
- `ollama` uses the local Ollama Python client and `format="json"`.
- SaaS outputs are practical baselines, not medical gold standards.
- MM-Skin is prompt-design inspiration only, not direct wound ground truth.
