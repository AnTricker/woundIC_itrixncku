# 620 Autoresearch Prompt Self-Improvement Plan

本文件重新定義 `autoresearch` 在本專案中的用法。

核心不是 ablation study，也不是 plugin architecture。核心是：

```text
agent-driven self-improvement loop for prompt engineering
```

也就是讓 agent 在固定資源限制下，根據既有實驗紀錄與評分結果，自動提出下一個 prompt 改良假設，產生小幅 prompt variant，跑 bounded experiment，分析結果，再產生下一輪改良方向。

Ablation test 仍然重要，但它只是 agent 用來定位原因的一種 experiment type，不是整個方法本身。

## 1. What To Borrow From Autoresearch

`karpathy/autoresearch` 的重點不是某個 training code，而是一個研究工作流：

```text
agent modifies one experiment target
-> runs a bounded experiment
-> evaluates with fixed metric
-> keeps or rejects
-> writes experiment log
-> repeats
```

原始 `autoresearch` 用在 single-GPU LLM training：

- agent 修改 `train.py`。
- 每次 train 固定 5 minutes。
- 用 validation metric 判斷是否進步。
- human 主要編輯 `program.md`，讓 agent 知道研究組織規則。

本專案不能直接照搬，因為我們目前不是 training model weights，而是在做：

```text
prompt / schema / domain context / translation / VQA evidence 的實驗改良
```

所以對應關係應該是：

| Autoresearch Element | Original Meaning | This Project Meaning |
|---|---|---|
| `program.md` | agent instruction / research org rule | prompt research rule / experiment policy |
| `train.py` | agent modifies training code | agent proposes prompt/schema/context variant |
| fixed 5-min run | comparable training budget | fixed image subset / local runtime budget |
| validation metric | model quality metric | BERTScore / schema valid / VQA / runtime metrics |
| keep or discard | accept code change | recommend keep / reject / human review |
| experiment log | overnight research trace | prompt self-improvement record |

## 2. Core Loop

本專案的 autoresearch loop 應設計成：

```text
load baseline prompt + previous records
-> agent identifies weakness
-> agent writes hypothesis
-> agent generates 1-3 prompt variants
-> run tiny local probe
-> evaluate metrics and evidence
-> agent analyzes why it improved or failed
-> decision: keep / reject / needs_human_review
-> write next hypothesis
```

第一版不需要真正「全自動 overnight」。先做到 deterministic、可回放、可人工 review 的 semi-auto loop。

## 3. Why This Matters For This Project

目前 captioning / wound description 的問題不是單純「分數低」，而是：

- local model output 有時 schema failed。
- lexical score 對同義詞與語序很敏感。
- BERTScore 需要 evidence 才能解釋。
- 中文 translation 是人工 review 的主要閱讀層，但 translation 本身也可能改變語意。
- prompt 太長時可能讓 local model 更不穩。
- schema 太複雜時 JSON validity 下降。
- healthcare domain hint 可能提升細節，也可能誘導 hallucination。

這些都不是一次改 prompt 能解決的問題。比較合理的方法是讓 agent 不斷提出「小假設」，用小規模資料快速測試，然後把結果變成下一輪的研究記憶。

## 4. Self-Improvement Unit

每一輪 autoresearch experiment 必須包含：

| Field | Meaning |
|---|---|
| `experiment_id` | unique id |
| `baseline_prompt_id` | compared baseline prompt |
| `variant_id` | new prompt/schema/context variant |
| `objective` | 本輪想改善什麼 |
| `hypothesis` | agent 認為為何這樣改會改善 |
| `changed_parts` | 實際改動的 prompt/schema/context parts |
| `dataset_scope` | probe / smoke / full |
| `model_id` | e.g. `gemma4:e2b`, `gemma4:26b`, `gemma4:12b` |
| `hardware_profile` | RTX 4090 / Ollama / runtime constraint |
| `metric_profile` | 本輪使用哪些 metrics |
| `result_summary` | 分數與 evidence 摘要 |
| `failure_analysis` | 為何失敗或可能失敗 |
| `decision` | keep / reject / needs_human_review |
| `next_hypothesis` | 下一輪可測方向 |

這個 unit 才是 autoresearch 的核心資料結構。

## 5. Agent Roles

第一版可以是一個 agent，但思考上拆成幾個 role 比較清楚。

| Role | Job |
|---|---|
| Research Planner | 根據 previous records 選本輪 objective |
| Prompt Mutator | 產生 prompt/schema/context variant |
| Runner | 跑 local probe / smoke |
| Evaluator | 計算 BERTScore、schema validity、VQA、runtime |
| Critic | 看 evidence，寫 failure analysis 和 next hypothesis |

不要一開始做 multi-agent。先讓單一 script / prompt 產出這些 sections，之後再拆。

## 6. Experiment Types

Ablation 是其中一種，不是全部。

| Type | Question |
|---|---|
| `ablation` | 拿掉某段 prompt 後，品質是否下降？這段是否真的有用？ |
| `mutation` | 重寫某段 instruction 後，是否更穩？ |
| `compression` | 縮短 prompt/schema 後，schema validity 或 runtime 是否改善？ |
| `domain_injection` | 加入 wound domain hint 後，細節是否更完整？ |
| `negative_constraint` | 加強不要臆測後，hallucination 是否下降？ |
| `field_reweighting` | 強調 visual_summary 後，是否比較接近人工閱讀需求？ |
| `translation_alignment` | 翻譯規則調整後，中英語意是否更一致？ |
| `vqa_feedback` | 用 caption 生成 MCQ 後，category answer 是否更可用？ |

所以 ablation 的位置應該是：

```text
self-improvement loop 的 diagnostic tool
```

不是：

```text
整個 autoresearch 方法
```

## 7. Metrics And Gates

每輪 experiment 不應只看一個分數。

| Metric | Why |
|---|---|
| AC BERTScore | English caption vs SaaS English reference |
| BD BERTScore | translated candidate vs translated reference，用於中文 review 對照 |
| visual_summary_only F1 | 避免 wound_features 固定短詞拉高 |
| full_caption_fields F1 | 檢查整體 structured caption |
| delta F1 | 檢查 full fields 是否造成 score inflation |
| schema valid rate | JSON 是否可被 pipeline 使用 |
| output token count | verbosity / local cost |
| runtime duration | local feasibility |
| VRAM peak | RTX 4090 是否能穩定跑 |
| GPU utilization | local run 是否吃滿或卡住 |
| VQA accuracy | caption 是否支援後續 wound category 問答 |
| human review flag | 分數看似高但文字不可信時標記 |

Decision rule 第一版只產生建議：

```text
keep
reject
needs_human_review
```

不要讓 agent 自動覆蓋正式 prompt。

## 8. Prompt / Schema / Context Search Space

agent 可改的 target 應該明確，不然會變成亂改。

| Target | Example |
|---|---|
| `task_instruction` | 要模型做 visual wound description |
| `safety_scope` | visual only, no diagnosis |
| `output_schema` | full schema / reduced schema / visual-summary schema |
| `visual_observation_guide` | body site, wound presence, visual summary |
| `wound_feature_guide` | edge, color, texture, exudate |
| `category_specific_hint` | abrasion / burn / bruise 等特徵提示 |
| `negative_constraints` | 不要補不存在內容 |
| `uncertainty_rule` | unknown / not observed 的寫法 |
| `translation_rule` | 繁中 + 保留英文 medical term |
| `vqa_question_rule` | 從 caption 轉成 MCQ 的規則 |

每輪 variant 最好限制在 1-2 個 changed targets，否則無法知道變好或變壞的原因。

## 9. Run Budget

因為目前目標是 local prompt engineering，不是 full benchmark，建議 budget：

| Stage | Use |
|---|---|
| `probe_3` | 3 images，確認 schema 和 obvious failure |
| `probe_10` | 每類 1-2 張，檢查類別差異 |
| `smoke` | 10% manifest，用於比較 prompt variant |
| `full` | 只有 stable candidate 才跑 |

RTX 4090 / Ollama baseline：

```text
GPU: RTX 4090
Runtime: Ollama local model
Candidate models: gemma4:e2b, gemma4:12b, gemma4:26b
```

每輪必須記錄：

- model tag
- prompt variant id
- image count
- input token count
- output token count
- total duration
- prefill time
- decode time
- runtime VRAM usage
- runtime GPU utilization
- schema valid / failed count

## 10. Record Format

建議未來輸出：

```text
runs/<run_id>/prompt_research/
  prompt_self_improvement_record.json
  prompt_self_improvement_report.md
  variants/
    <variant_id>.md
```

JSON 概念：

```json
{
  "experiment_id": "prompt_research_001",
  "baseline_prompt_id": "simple_v1",
  "variant_id": "simple_v1_mutation_visual_summary_001",
  "objective": "Improve visual_summary specificity without hurting schema validity.",
  "hypothesis": "A shorter visual observation guide will reduce generic text and improve visual_summary_only F1.",
  "changed_parts": ["visual_observation_guide"],
  "dataset_scope": "probe_10",
  "model_id": "gemma4:26b",
  "metric_profile": ["AC", "BD", "visual_summary_only", "schema_valid_rate", "runtime"],
  "result_summary": {},
  "failure_analysis": "",
  "decision": "needs_human_review",
  "next_hypothesis": ""
}
```

## 11. Related Methods To Borrow

這些方法不是要直接套 framework，而是提供 design hint。

| Method | Useful Idea For This Project |
|---|---|
| Autoresearch | 固定 budget、agent 自動改一個 target、跑實驗、看 metric、記錄結果、重複 |
| OPRO | 把 previous solutions + scores 放回 prompt，讓 LLM 產生下一批 candidate prompts |
| TextGrad | 把 textual feedback 當成 gradient-like signal，指向應修改的 prompt component |
| DSPy / MIPRO | 把 prompt pipeline 模組化，用 metric 編譯/搜尋較好的 instruction 或 demonstrations |
| Environment-grounded prompt optimization | 用 rollout trace / behavior analysis 找 failure mode，再 mutation prompt |

對本專案最實用的融合方式：

```text
Autoresearch loop structure
+ OPRO-style previous score memory
+ TextGrad-style failure feedback
+ DSPy-style modular prompt target
+ local smoke metric gate
```

## 12. Local Model As Agent

實作時，local model 不能只被當成 caption generator。它也可以被包成一個簡化版 agent。

這裡的 agent 不需要像 Codex / Claude 那樣一開始就有完整 IDE 操作能力。第一版只需要：

```text
read context
-> reason over records
-> propose next experiment
-> write structured JSON / Markdown
```

也就是把 `gemma4:*` 當成 planner / critic / prompt mutator，而不是讓它直接 uncontrolled 地改 repo。

建議第一版 agent interface：

| Component | Meaning |
|---|---|
| `context_loader` | 讀取 SDD、score report、schema failures、BERTScore evidence |
| `agent_prompt` | 告訴 local model 它要提出下一輪 prompt improvement |
| `ollama_runner` | 呼叫 local Ollama model，例如 `gemma4:26b` |
| `structured_output_parser` | 要求輸出固定 JSON schema |
| `proposal_writer` | 寫出 proposal `.json` / `.md` |
| `human_gate` | 人工確認後才真的跑 experiment 或採用 prompt |

第一版 local agent 只產出 proposal，不直接執行 destructive action：

```text
allowed:
  - summarize previous results
  - identify weak fields
  - propose prompt variants
  - propose what extra evidence is needed
  - write experiment plan

not allowed in first version:
  - overwrite production prompt
  - delete run outputs
  - auto-commit code
  - run large full dataset without approval
```

最小化 command 概念：

```bash
python scripts/prompt_research_agent.py \
  --run-dir runs/<run_id> \
  --model gemma4:26b \
  --baseline-prompt prompts/simple_v1.md
```

此 command 的用途：

```text
用 local Ollama model 讀取目前 run 的結果，產生下一輪 prompt self-improvement proposal。
```

輸出建議：

```text
runs/<run_id>/prompt_research/proposals/<proposal_id>.json
runs/<run_id>/prompt_research/proposals/<proposal_id>.md
```

重點：

```text
local model is the reasoning agent
pipeline scripts are the tools
human remains the approval gate
```

## 13. Survey / Knowledge Extension Queue

Autoresearch 不一定只能「改 prompt 再跑分數」。它也可以幫我們辨識：

```text
目前缺哪種外部知識或方法資料？
```

這對本專案很重要，因為 prompt engineering 可能卡在不是 prompt wording，而是缺少某種 knowledge source，例如：

- wound description terminology 不夠完整。
- 不知道哪種 semantic metric 比 BERTScore 更適合。
- 不確定 VQA dataset 應該如何設計 distractors。
- 不知道 medical image captioning 常用哪些 evaluation。
- 不知道 schema reduction 是否有類似方法。
- 不知道 Gemma / Gemini 在 multilingual captioning 的已知限制。

因此 autoresearch loop 可多輸出一個 `survey_queue`。

### 13.1 Survey Queue Format

```json
{
  "survey_queue": [
    {
      "topic": "semantic metric for medical image captions",
      "why_needed": "Current lexical metrics are low and BERTScore needs calibration.",
      "expected_use": "Decide whether to add LLM-as-judge or domain embedding score.",
      "priority": "high",
      "manual_action": "User searches papers / docs and adds findings to knowledge base."
    }
  ]
}
```

### 13.2 Human-In-The-Loop Research Flow

建議流程：

```text
agent detects knowledge gap
-> writes survey_queue item
-> human searches manually
-> human adds notes / sources to local knowledge base
-> agent uses updated knowledge in next proposal
```

這樣 autoresearch 不是假裝自己能完全自動 research，而是變成：

```text
agent identifies what information is missing
human supplies trusted new information
agent incorporates it into next prompt experiment
```

### 13.3 Suggested Local Knowledge Base

未來可使用簡單 Markdown / JSONL，不要一開始上複雜 vector DB。

```text
docs/research_notes/
  metrics.md
  wound_captioning.md
  prompt_optimization.md
  vqa_dataset.md
```

或 machine-readable：

```text
runs/<run_id>/prompt_research/knowledge_queue.json
```

每筆 notes 應該至少有：

| Field | Meaning |
|---|---|
| `topic` | 研究主題 |
| `source` | paper / docs / manual note |
| `summary` | human 摘要 |
| `usable_in_prompt` | 是否可轉成 prompt context |
| `usable_in_metric` | 是否可轉成 evaluation method |
| `risk` | 是否可能造成 hallucination / bias |

## 14. Portable Interface Is Secondary

Decoupling 仍然需要，但它是為了讓 self-improvement loop 可重用，不是本週核心。

可插拔 interface：

| Plugin Point | Interface |
|---|---|
| Healthcare knowledge | wound_type -> visual features -> prompt context |
| VQA | caption -> MCQ -> answer -> accuracy |
| Translation | English caption -> zh-TW caption |
| Schema | schema version -> validator -> field cost |
| Model runtime | model tag -> local/SaaS runner -> runtime metadata |
| Metric | outputs -> score report -> decision gate |

原則：

```text
loop stays the same
task modules can change
```

## 15. First Implementation Boundary

本文件目前仍是 SDD / design，不是 implementation request。

第一個可實作版本應該很小：

1. 讀取一份 baseline prompt。
2. 讀取既有 scores / schema failures / BERTScore evidence。
3. 呼叫 local Ollama model 產生一份 prompt improvement proposal。
4. 同時產生 `survey_queue`，列出需要人工補充的資料。
5. 只跑 `probe_3` 或 `probe_10`。
6. 產生 `prompt_self_improvement_report.md`。
7. 不自動改正式 prompt。

不要一開始做：

- overnight fully autonomous loop。
- 自動改 production prompt。
- 多 agent orchestration。
- 大量 Gemini call。
- full dataset 搜尋。

## 16. Acceptance Criteria

這份設計完成後，下一階段 implementation 應能回答：

1. 這輪 agent 想改善什麼？
2. 它為什麼認為這樣改會改善？
3. 它改了 prompt 的哪一段？
4. 它跑了哪些 images？
5. 分數如何變化？
6. evidence text 是否支持分數？
7. schema / runtime 是否變穩或變差？
8. 下一輪應該測什麼？
9. 是否值得人工 review 或採用？
10. 它認為目前缺哪些外部資料或方法 survey？
11. 人工補充的新資料如何進入下一輪 prompt experiment？

## 17. Source Notes

- `karpathy/autoresearch`: <https://github.com/karpathy/autoresearch>
- OPRO, Large Language Models as Optimizers: <https://arxiv.org/abs/2309.03409>
- TextGrad: Automatic Differentiation via Text: <https://arxiv.org/abs/2406.07496>
- DSPy: Compiling Declarative Language Model Calls into Self-Improving Pipelines: <https://arxiv.org/abs/2310.03714>
