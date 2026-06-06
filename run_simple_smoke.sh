#!/usr/bin/env bash
set -euo pipefail

IMAGE_DIR="${IMAGE_DIR:-images}"
LOCAL_MODEL="${LOCAL_MODEL:-gemma4:26b}"
SAAS_MODEL="${SAAS_MODEL:-gemini-3.1-flash-lite}"
SMOKE_PERCENT="${SMOKE_PERCENT:-10}"
SEED="${SEED:-42}"
RUN_DATE="${RUN_DATE:-$(date +%Y%m%d)}"
MODEL_SLUG="${LOCAL_MODEL//:/_}"
RUN_DIR="${RUN_DIR:-runs/${RUN_DATE}_${MODEL_SLUG}}"

# gemini-3.1-flash-lite is planned at 15 RPM, so 4 sec/request is the safe minimum.
SAAS_REQUEST_DELAY_SEC="${SAAS_REQUEST_DELAY_SEC:-4}"
SAAS_MAX_RETRIES="${SAAS_MAX_RETRIES:-3}"
SAAS_BACKOFF_BASE_SEC="${SAAS_BACKOFF_BASE_SEC:-10}"

echo "[1/5] Prepare 10% smoke manifest"
python scripts/prepare.py \
  --image-dir "$IMAGE_DIR" \
  --run-dir "$RUN_DIR" \
  --smoke-percent "$SMOKE_PERCENT" \
  --exclude-dirs MM-SkinQA \
  --seed "$SEED"

echo "[2/5] Run local simple smoke"
python scripts/generate_local.py \
  --run-dir "$RUN_DIR" \
  --scopes smoke \
  --modes simple \
  --local-model "$LOCAL_MODEL" \
  --workers 1

# echo "[3/5] Run Gemini simple smoke"
# python scripts/generate_saas.py \
#   --run-dir "$RUN_DIR" \
#   --scopes smoke \
#   --modes simple \
#   --saas-provider gemini \
#   --saas-model "$SAAS_MODEL" \
#   --workers 1 \
#   --saas-request-delay-sec "$SAAS_REQUEST_DELAY_SEC" \
#   --saas-max-retries "$SAAS_MAX_RETRIES" \
#   --saas-backoff-base-sec "$SAAS_BACKOFF_BASE_SEC"

echo "[4/5] Score simple smoke"
python scripts/score.py \
  --run-dir "$RUN_DIR" \
  --reference-mode simple \
  --score-scopes smoke \
  --saas-simple-baseline-dir runs/saas_simple_baseline

echo "[5/5] Generate charts"
python scripts/charts.py \
  --run-dir "$RUN_DIR"

echo "Done."
echo "Read: $RUN_DIR/scores.md"
echo "Read: $RUN_DIR/experiment_record.json"
