# itriIC

Wound image VLM prompt-engineering and evaluation pipeline.

這個 repo 目前的核心目標是：用同一批 wound images，同時跑 local VLM 與 SaaS VLM，並比較 `simple` prompt 和 `full` multi-stage prompt 的效果。重點不是醫療診斷，而是 visual caption / VQA output 的穩定性、feature coverage、schema validity、hallucination control，以及 local model 與 SaaS reference 的接近程度。

## Repo Structure

```text
.
├── pipeline.py                  # evaluation pipeline CLI entry point
├── main.py                      # earlier local inference entry point
├── prompts/                     # prompt templates and category prompts
├── docs/
│   ├── evaluation_pipeline.md   # detailed evaluation command guide
│   └── prompt_design_analysis.md# prompt design rationale
├── reference/                   # VLM-AutoDrive / MM-Skin references
├── images/                      # local dataset, ignored by git
├── runs/                        # generated manifests, outputs, scores, ignored by git
├── requirements.txt
├── .env.example
└── wound_schema.json
```

## Environment

Use Python 3.10+ in a virtual environment or conda environment before running anything.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

這三條指令的意思：

- `python -m venv .venv`：在 repo 裡建立 Python virtual environment。
- `source .venv/bin/activate`：啟用這個環境，後續 `python` / `pip` 都會用 `.venv`。
- `python -m pip install -r requirements.txt`：安裝 pipeline 需要的套件。

如果你在 server 上已經用 conda，例如 `(itriIC)`，可以不用再建 `.venv`，但要確認套件是裝在目前啟用的 env 裡。

## API Keys

API key 放在 `.env`，不要放在 command 裡。

```bash
cp .env.example .env
```

然後編輯 `.env`：

```text
GEMINI_API_KEY=your_gemini_api_key
OPENAI_API_KEY=your_openai_api_key
```

注意：Gemini / ChatGPT Plus web subscription 不等於 API key。pipeline 需要的是 API credential。

## Dataset Layout

預期資料夾格式：

```text
images/
├── Abrasions/
├── Bruises/
├── Burns/
├── Cut/
├── Ingrown_nails/
├── Laceration/
├── Stab_wound/
└── MM-SkinQA/
```

`MM-SkinQA` 預設應排除，因為它是 reference/vocabulary source，不是目前 wound category evaluation set。

## Quickstart

先建立 manifest，抽 10% 做 smoke test：

```bash
python -m pipeline prepare \
  --image-dir images \
  --run-dir runs/dev \
  --smoke-percent 10 \
  --exclude-dirs MM-SkinQA \
  --seed 42
```

跑 local + SaaS 的 smoke generation：

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

確認 SaaS baseline 品質後，再算分：

```bash
python -m pipeline score \
  --run-dir runs/dev \
  --reference-mode full \
  --score-scopes smoke
```

完整指令說明在 [docs/script_command_guide.md](docs/script_command_guide.md)。

## Simple-only Formal Generation

全資料使用 `formal` scope；`simple` 每張影像只執行一次 caption model call。

Local 全量 simple generation：

```bash
python scripts/generate_local.py \
  --run-dir runs/20260816_gemma3_4b_formal \
  --scopes formal \
  --modes simple \
  --local-model gemma3:4b \
  --workers 1
```

Gemini simple-only probe（從指定 scope 選一張影像）：

```bash
python scripts/gemini_probe.py \
  --run-dir runs/20260816_gemma3_4b_formal \
  --scopes formal \
  --modes simple \
  --seed 42 \
  --saas-model gemini-3.1-flash-lite \
  --saas-request-delay-sec 5 \
  --saas-max-retries 5 \
  --saas-backoff-base-sec 5 \
  --saas-rpm 15 \
  --saas-rpd 500 \
  --saas-tpm 250000
```

`gemini_probe.py` 預設模式是 `simple`；仍可明確指定 `full` 或
`simple,full`。`--saas-rpm`、`--saas-rpd`、`--saas-tpm` 應填入目前
Google AI Studio 專案顯示的實際 quota。內建值只用於 planning，並非帳戶保證值。

RPM/TPM 行為：

- `--saas-request-delay-sec` 是實際 request pacing；15 RPM 建議設為 5 秒，保留緩衝。
- RPM/RPD/TPM 參數供 probe projection 與安全檢查使用。
- TPM 目前沒有 token-bucket 強制限流；以 Gemini `usageMetadata` 的 input token 推估。
- SaaS generation 建議 `--workers 1`，避免並行 request 突破 pacing 假設。
- Gemini quota 以 project 為單位且會隨 model/tier 改變，請查看 [Google Gemini rate limits](https://ai.google.dev/gemini-api/docs/rate-limits)。

## Prompt Modes

`simple` 是 single-stage baseline：

- 一張圖一次 model call。
- 直接輸出 schema-compatible JSON。
- 成本低，速度快。
- 缺點是 feature coverage、negative reporting、hallucination control 較弱。

`full` 是 multi-stage prompt pipeline：

- `caption_template.md`：產生 structured wound caption。
- `auxiliary_vqa_template.md`：固定 wound visual QA 補 coverage。
- `self_check_template.md`：檢查 JSON、診斷越界、尺寸 hallucination、無 evidence 推論。

詳細設計邏輯在 [docs/prompt_design_analysis.md](docs/prompt_design_analysis.md)。

## Evaluation Concept

目前不是 train/validate split。

現在是：

- `smoke`：抽樣一部分資料，快速檢查 pipeline、prompt、SaaS baseline 品質。
- `formal`：全資料正式生成與評分。

要算 local vs SaaS 的 BLEU / ROUGE / METEOR / CIDEr-lite，兩邊必須對同一張圖都有 output。所以 local 和 SaaS 不是拆 dataset，而是在同一個 scope 裡生成對應輸出。

2x2 ablation：

| Provider | Simple | Full |
|---|---|---|
| local | local + simple | local + full |
| SaaS | SaaS + simple | SaaS + full |

## Outputs

常見輸出位置：

```text
runs/dev/
├── manifest.jsonl
├── split_summary.json
├── experiment_record.json
├── outputs/
├── probes/
├── scores.json
├── scores.md
└── <run-name>_record_summary.md
```

`runs/` 是 generated output，預設不進 git。

生成 run-local metadata 報告：

```bash
python scripts/metadata.py --run-dir runs/20260816_gemma3_4b_formal
```

輸出為 `runs/20260816_gemma3_4b_formal/20260816_gemma3_4b_formal_record_summary.md`。
可在 generation/probe 後直接執行；若尚無 `scores.json`，score 欄位會顯示
`not recorded`。

## Practical Notes

- `Ctrl+C` 可以中止長時間 generation。
- 重新跑同一個 `run-dir` 可能覆蓋同名 output；如果要保留舊結果，換一個 `--run-dir`。
- 速度慢時先跑 `--modes simple` 或降低 `--smoke-percent`。
- SaaS 成本高時先只跑 `--saas-scopes smoke`，確認 baseline 品質後再跑 `formal`。
- local model 很慢時，優先減少 `full`，因為 `full` 每張圖約 3 次 model call；同時跑 `simple,full` 約 4 次 call。

## Troubleshooting

如果出現 import error，例如 `ModuleNotFoundError`，先確認目前 Python 是 env 裡的 Python：

```bash
which python
python -m pip install -r requirements.txt
```

如果 conda env 仍吃到 user-site package，可暫時用：

```bash
PYTHONNOUSERSITE=1 python -m pipeline prepare --image-dir images --run-dir runs/dev
```

這代表執行 Python 時忽略 user-site packages，避免系統或使用者層級套件污染目前 env。
