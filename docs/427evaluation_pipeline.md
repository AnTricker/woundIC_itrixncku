# Wound VLM Evaluation Pipeline

## 0. Core Concept

本專案目前不是 training pipeline，所以沒有 train/validate split。

現在 manifest 使用兩種 scope：

- `smoke`: 隨機抽樣子集，預設 10%，用來快速檢查整個 pipeline、prompt output、SaaS reference 品質。
- `formal`: 100% supported wound dataset，用來正式跑分。

被抽到 `smoke` 的圖片同時也屬於 `formal`。也就是：

```text
smoke 是 formal 的 subset
formal 是完整資料
```

程式入口在 repo root：

```bash
python -m pipeline ...
```

---

## 1. API Key Setup

API key 放在 repo root 的 `.env`，不要寫在 command 裡。

```bash
cp .env.example .env
```

這條指令意思：

- `cp`: copy file。
- `.env.example`: repo 提供的範本，不含真實 key，可以 commit。
- `.env`: 你的本機 secret file，已被 `.gitignore` 忽略，不應 commit。

接著編輯 `.env`：

```env
GEMINI_API_KEY=your_gemini_api_key
OPENAI_API_KEY=your_openai_api_key
```

彈性設定：

- 只用 Gemini 時，只填 `GEMINI_API_KEY` 即可。
- 只用 OpenAI 時，只填 `OPENAI_API_KEY` 即可。
- 兩個都填，跑 command 時用 `--saas-provider` 決定實際用哪個。

如果你不想用 repo root `.env`，可以指定其他 env file：

```bash
python -m pipeline --env-file /path/to/.env generate-saas ...
```

參數說明：

- `--env-file /path/to/.env`
  - 指定 API key 檔案位置。
  - 不指定時，預設讀 repo root 的 `.env`。

---

## 2. Prepare Manifest

Prepare step 會掃描 `images/`，根據第一層資料夾推斷 wound category，排除 `MM-SkinQA`，然後產生 manifest。

預期資料夾：

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

逐條說明：

- `python -m pipeline prepare`
  - 執行 repo root 的 `pipeline.py`。
  - `prepare` 是建立 manifest 的子命令。
  - 只掃資料、抽樣、寫 manifest，不會呼叫模型。

- `--image-dir images`
  - 指定原始圖片資料夾。
  - 如果圖片在其他地方，例如 `/data/itri/images`，改成 `--image-dir /data/itri/images`。

- `--run-dir runs/dev`
  - 指定本次實驗輸出資料夾。
  - manifest、model outputs、scores 都會放在這裡。
  - 建議重要實驗換一個 run dir，例如 `runs/0427_gemma4_e2b_smoke`。

- `--smoke-percent 10`
  - 從完整 wound dataset 隨機抽 10% 當 smoke subset。
  - 不是 split 掉另外 90%；全部資料仍然是 `formal`。
  - 想更快測試可設 `--smoke-percent 3`。
  - 想抽更多檢查 prompt 穩定性可設 `--smoke-percent 20`。

- `--exclude-dirs MM-SkinQA`
  - 掃描圖片時排除 `images/MM-SkinQA`。
  - 預設就是 `MM-SkinQA`，寫出來只是讓 command 更清楚。
  - 若要排除多個資料夾，用 comma，例如 `--exclude-dirs MM-SkinQA,bad_cases,tmp`。

- `--seed 42`
  - 控制隨機抽樣。
  - 同樣 seed + 同樣資料集會抽到同一批 smoke。
  - 想重新抽一批 smoke，就換 seed，例如 `--seed 123`。

Outputs:

- `runs/dev/manifest.jsonl`
  - 每張 supported wound image 一行。
  - 每行有 `image`、`image_name`、`category`、`scopes`。

- `runs/dev/excluded_manifest.jsonl`
  - 不支援或被排除的圖片紀錄。

- `runs/dev/split_summary.json`
  - 實驗摘要，包含 `formal` / `smoke` 數量、各 category 數量。

範例 summary：

```json
{
  "counts": {
    "formal": 560,
    "smoke": 56
  }
}
```

---

## 3. Smoke Test Generation

Smoke test 用 10% 子集檢查 pipeline、prompt、SaaS output 品質。

Command:

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

逐條說明：

- `python -m pipeline generate`
  - 執行 generate 子命令。
  - 會同時支援 local provider 和 SaaS provider。
  - 會依照 manifest 選出指定 scope 的圖片。

- `--run-dir runs/dev`
  - 指向前面 `prepare` 產生 manifest 的 run dir。
  - 必須和 prepare 用同一個資料夾，否則找不到 manifest。

- `--local-scopes smoke`
  - local Ollama 只跑 `smoke` subset。
  - 正式全量時改成 `--local-scopes formal`。

- `--saas-scopes smoke`
  - SaaS VLM 只跑 `smoke` subset。
  - 要算 local vs SaaS score，local 和 SaaS 必須在同一個 scope 都有 output。
  - 正式全量時改成 `--saas-scopes formal`。

- `--modes simple,full`
  - 同時跑 simple prompt 和 full prompt。
  - `simple` 每張圖約 1 次 model call。
  - `full` 每張圖約 3 次 model call：caption、auxiliary VQA、self-check。
  - 如果覺得太慢，可以先只跑 `--modes simple`，確認 OK 後再補 `--modes full`。

- `--local-model gemma4:e2b`
  - 指定 Ollama local model name。
  - 必須是 `ollama list` 裡存在的模型。
  - 可換成 `--local-model qwen3.5` 或 `--local-model llava:7b`。

- `--saas-provider gemini`
  - 指定 SaaS provider。
  - 可選 `gemini` 或 `openai`。

- `--saas-model gemini-2.5-flash`
  - 指定 SaaS model。
  - Gemini 範例：`--saas-model gemini-2.5-flash`。
  - OpenAI 範例：`--saas-provider openai --saas-model gpt-4.1-mini`。

- `--parallel-providers`
  - 讓 local 和 SaaS 兩條 provider pipeline 並行。
  - 可以省時間。
  - 如果 GPU 很滿、API rate limit 不穩，可以拿掉這個 flag，改成順序執行。

- `--workers 1`
  - 每個 provider 內部同時處理幾個 job。
  - local Ollama 通常先用 `1` 最穩。
  - SaaS 如果 rate limit 允許，可以試 `--workers 2`。
  - local GPU VRAM 不夠時不要亂加 workers。

Outputs:

```text
runs/dev/outputs/local/simple/*.json
runs/dev/outputs/local/full/*.json
runs/dev/outputs/saas/simple/*.json
runs/dev/outputs/saas/full/*.json
```

---

## 4. Score Smoke

Command:

```bash
python -m pipeline score \
  --run-dir runs/dev \
  --reference-mode full \
  --score-scopes smoke
```

逐條說明：

- `python -m pipeline score`
  - 執行 score 子命令。
  - 不會呼叫模型。
  - 只讀已存在的 output JSON。

- `--run-dir runs/dev`
  - 指定要評分的 run dir。

- `--reference-mode full`
  - 指定 SaaS reference 使用哪個 mode。
  - 通常用 `full`，因為 full prompt coverage 比 simple 好。
  - 如果想看 local output 對齊 SaaS simple 的程度，可改 `--reference-mode simple`。

- `--score-scopes smoke`
  - 只評分 smoke subset。
  - 正式全量改成 `--score-scopes formal`。

Outputs:

- `runs/dev/scores.json`
  - machine-readable score。

- `runs/dev/scores.csv`
  - table format，方便丟 spreadsheet。

- `runs/dev/scores.md`
  - human-readable 2x2 ablation table。

---

## 5. Formal Run

Smoke 確認 OK 後，正式跑 100% supported wound dataset。

Command:

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

和 smoke command 的差異：

- `--local-scopes formal`
  - local 跑完整資料。

- `--saas-scopes formal`
  - SaaS 跑完整資料。

注意：

- formal + `simple,full` 成本高，因為每張圖總共約 4 次 call。
- 如果正式跑太慢，可先 `--modes simple`，再補 `--modes full`。

Score formal:

```bash
python -m pipeline score \
  --run-dir runs/dev \
  --reference-mode full \
  --score-scopes formal
```

---

## 6. One Command Smoke Run

這個 command 會一次做：

1. prepare manifest
2. generate local smoke
3. generate SaaS smoke
4. 停在 scoring 前讓你人工檢查 output

Command:

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

逐條說明：

- `python -m pipeline run-all`
  - 一次串起 prepare + generate + optional score。

- `--image-dir images`
  - 原始圖片資料夾。

- `--run-dir runs/dev`
  - 本次實驗資料夾。

- `--smoke-percent 10`
  - 抽 10% 當 smoke。

- `--local-scopes smoke`
  - local 只跑 smoke。

- `--saas-scopes smoke`
  - SaaS 只跑 smoke。

- `--modes simple,full`
  - simple/full 都跑。

- `--local-model gemma4:e2b`
  - Ollama local model。

- `--saas-provider gemini`
  - SaaS provider。

- `--saas-model gemini-2.5-flash`
  - SaaS model。

- `--parallel-providers`
  - local/SaaS 並行。

- `--stop-before-score`
  - generate 完後停止。
  - 用來先人工看 SaaS reference 是否夠好。
  - 如果拿掉這個 flag，run-all 會直接 score。

接著 score：

```bash
python -m pipeline score \
  --run-dir runs/dev \
  --reference-mode full \
  --score-scopes smoke
```

---

## 7. Scoring Mechanism

Semantic scoring 需要同一張圖同時有 local output 和 SaaS output。

例如：

```text
runs/dev/outputs/local/full/abrasions (1).json
runs/dev/outputs/saas/full/abrasions (1).json
```

如果沒有 paired output，BLEU/ROUGE/METEOR/CIDEr-lite 的 `paired_count` 會是 0。

Text similarity metrics：

- `BLEU-4`
  - 看 n-gram precision。
  - 對 wording 很敏感。

- `ROUGE-L`
  - 看 longest common subsequence。
  - 比 BLEU 稍微容忍語序差異。

- `METEOR-lite`
  - 目前是簡化版 token overlap F-score。
  - 用於快速 iteration。

- `CIDEr-lite`
  - 目前是簡化版 n-gram cosine similarity。
  - 用於快速 iteration。

Behavior metrics：

- `schema_valid_rate`
  - JSON 是否符合 `wound_schema.json`。

- `required_field_completion_rate`
  - 必填欄位是否有填。

- `not_observed_usage_rate`
  - 是否有使用 negative reporting。

- `no_ruler_size_compliance_rate`
  - 沒有 ruler/calibration object 時是否避免尺寸 hallucination。

- `safety_scope_compliance_rate`
  - 是否保留 `"Visual description only; not a medical diagnosis."`

---

## 8. Provider Notes

- `gemini`
  - 使用 REST `models/{model}:generateContent`。
  - 圖片用 inline base64 data。

- `openai`
  - 使用 Responses API。
  - 圖片用 base64 image data URL。

- `ollama`
  - 使用本地 Ollama Python client。
  - 使用 `format="json"` 要求 JSON output。

- SaaS outputs
  - 是 practical reference，不是 medical gold standard。

- MM-Skin
  - 是 prompt-design inspiration，不是 wound ground truth。
