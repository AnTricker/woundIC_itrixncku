# Wound VLM Evaluation Pipeline

## Core Concept

There is no train/validate split in this project. We are not training a model.

The dataset manifest now uses two scopes:

- `smoke`: a random subset, default 10%, for quick pipeline and prompt checks.
- `formal`: 100% of the supported wound dataset.

Images selected for `smoke` are also part of `formal`. This lets you first test the whole pipeline cheaply, inspect SaaS VLM quality, then run the same commands on the full dataset.

The code lives at repo root:

```bash
python -m pipeline ...
```

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
python -m pipeline --env-file /path/to/.env generate-saas ...
```

## 1. Prepare Manifest

The prepare step scans `images/`, infers wound category from the first folder name, excludes `MM-SkinQA` by default, and writes a manifest. It does not copy images.

Expected dataset layout:

```text
images/
├── Abrasions/
├── Bruises/
├── Burns/
├── Cut/
├── Ingrown_nails/
├── Laceration/
├── MM-SkinQA/
└── Stab_wound/
```

Command:

```bash
python -m pipeline prepare \
  --image-dir images \
  --run-dir runs/dev \
  --smoke-percent 10 \
  --exclude-dirs MM-SkinQA \
  --seed 42
```

Outputs:

- `runs/dev/manifest.jsonl`
- `runs/dev/excluded_manifest.jsonl`
- `runs/dev/split_summary.json`

`split_summary.json` will show counts like:

```json
{
  "counts": {
    "formal": 560,
    "smoke": 56
  }
}
```

## 2. Smoke Test

Run both local and SaaS VLM on the 10% smoke subset. This checks the full pipeline and prompt behavior before spending time/API cost on 100%.

```bash
python -m pipeline generate \
  --run-dir runs/dev \
  --local-scopes smoke \
  --saas-scopes smoke \
  --modes simple,full \
  --local-model gemma4:e2b \
  --saas-provider gemini \
  --saas-model gemini-2.5-flash \
  --parallel-providers \
  --workers 1
```

Outputs:

- `runs/dev/outputs/local/simple/*.json`
- `runs/dev/outputs/local/full/*.json`
- `runs/dev/outputs/saas/simple/*.json`
- `runs/dev/outputs/saas/full/*.json`

Score smoke:

```bash
python -m pipeline score \
  --run-dir runs/dev \
  --reference-mode full \
  --score-scopes smoke
```

Review:

- `runs/dev/scores.md`
- `runs/dev/outputs/saas/full/`
- `runs/dev/outputs/local/full/`

## 3. Formal Run

After smoke results look acceptable, run the full dataset with `formal`.

```bash
python -m pipeline generate \
  --run-dir runs/dev \
  --local-scopes formal \
  --saas-scopes formal \
  --modes simple,full \
  --local-model gemma4:e2b \
  --saas-provider gemini \
  --saas-model gemini-2.5-flash \
  --parallel-providers \
  --workers 1
```

Score formal:

```bash
python -m pipeline score \
  --run-dir runs/dev \
  --reference-mode full \
  --score-scopes formal
```

## One Command Smoke Run

This prepares the manifest, runs local + SaaS on the smoke subset, then stops before score so you can inspect outputs.

```bash
python -m pipeline run-all \
  --image-dir images \
  --run-dir runs/dev \
  --smoke-percent 10 \
  --local-scopes smoke \
  --saas-scopes smoke \
  --modes simple,full \
  --local-model gemma4:e2b \
  --saas-provider gemini \
  --saas-model gemini-2.5-flash \
  --parallel-providers \
  --stop-before-score
```

Then:

```bash
python -m pipeline score \
  --run-dir runs/dev \
  --reference-mode full \
  --score-scopes smoke
```

## Scoring Mechanism

Semantic scoring needs paired outputs for the same image.

For `abrasions (1).jpg`, these must both exist:

```text
runs/dev/outputs/local/full/abrasions (1).json
runs/dev/outputs/saas/full/abrasions (1).json
```

Then the pipeline compares local text against SaaS reference text using:

- BLEU-4
- ROUGE-L
- METEOR-lite
- CIDEr-lite

It also computes behavior metrics that do not require text pairing:

- schema valid rate
- required field completion rate
- not-observed usage rate
- no-ruler size compliance rate
- safety-scope compliance rate

## Provider Notes

- `gemini` uses REST `models/{model}:generateContent` with inline image data.
- `openai` uses the Responses API with base64 image data URLs.
- `ollama` uses the local Ollama Python client and `format="json"`.
- SaaS outputs are practical references, not medical gold standards.
- MM-Skin is prompt-design inspiration only, not direct wound ground truth.
