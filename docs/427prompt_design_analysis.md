# Prompt Design Analysis

## 0. 文件目的

這份文件說明目前 `itriIC` 的 prompt engineering 設計來源與實作邏輯。

本專案目前不是要讓 VLM 自由描述傷口，而是把 wound image captioning 拆成固定、可驗證、可比較的任務：

- 從 `VLM-AutoDrive` 借 prompt 結構。
- 從 `MM-Skin` caption/VQA 檔案借 dermatology vocabulary 與 question pattern。
- 設計 `simple` 與 `full` 兩種 prompt mode。
- 讓 local VLM 與 SaaS VLM 可以在同一套 JSON schema 下做 2x2 ablation。

核心目標：

- 提高 wound feature coverage。
- 降低 prompt-induced hallucination。
- 強制 `not observed` / `uncertain` 回報。
- 限制模型只做 visual description，不做 diagnosis。
- 讓 output 可以被 pipeline 自動評分。

---

## 1. Reference Analysis

## 1.1 VLM-AutoDrive Prompt Appendix 的邏輯

`reference/VLM-AutoDrive.md` 整理的是 autonomous driving VLM 的 prompt appendix。雖然 domain 是 driving event，不是 wound image，但它的 prompt design 很適合移植到醫療影像 annotation。

它最重要的想法是：

```text
不要只叫模型 describe image/video。
要把視覺理解拆成固定 dimensions、mandatory checks、reasoning QA、structured output。
```

這能避免模型只講它最顯眼的東西，漏掉評估上真正重要的細節。

### 1.1.1 Understanding VQA

`Understanding VQA` 的邏輯：

- 模型被要求根據 visual content 產生多個 QA triplets。
- 每題要覆蓋不同 dimension。
- 每題有 options 與 single answer。
- 問題不能只是表面描述，要能反映 deep understanding。
- Output 是固定結構。

在 VLM-AutoDrive 中，它要求覆蓋：

- weather / visibility / light condition
- road and environment
- vehicle motion
- impact side
- pedestrian/cyclist behavior
- vehicle role/type
- traffic density / scene context
- collision / near collision / normal driving

這個邏輯移植到 wound image 後，對應成：

- body site / location
- wound presence
- shape pattern
- edge / margin
- wound bed
- color
- texture
- exudate / bleeding / crust / pus
- periwound skin
- depth cue
- debris / foreign material
- necrosis / eschar
- measurement tool visibility
- uncertainty / not observed

我們借的是 `dimension coverage`，不是 driving 內容。

### 1.1.2 Mandatory Dimension

VLM-AutoDrive 有一個 mandatory question：

```text
這是 collision、near collision，還是 normal driving？
```

這種設計代表：某些 dimension 對任務結果太重要，不能讓模型自由決定要不要回答。

在 wound prompt 裡，我們對應成 mandatory checks：

- Is the epidermis visibly broken?
- Is there visible exudate / bleeding / crust / pus?
- Is blistering or bullae visible?
- Is necrosis or eschar visible?
- Is there a ruler or calibration card?
- Which category-expected features are not observed?

這些問題會在 `Auxiliary VQA Prompt` 裡固定出現。

### 1.1.3 Reasoning VQA

`Reasoning VQA` 的邏輯：

- 問題需要多步推理。
- 但推理必須依附 visual details。
- 模型不能憑常識或 metadata 亂推。

這對 wound task 很重要，因為模型很容易看到 category label 後直接說：

```text
This is a second-degree burn.
```

但這種 statement 如果沒有 visual evidence，就不可靠。

所以我們在 prompt 裡轉成：

- 不要求模型輸出完整 chain-of-thought。
- 但每個 category-level inference 都要有 `observed_supporting_features`。
- 無法從影像判斷的東西放到 `cannot_determine`。

也就是：

```text
保留 evidence-bound reasoning，但不外露冗長推理。
```

### 1.1.4 Thinking Trace

`Thinking Trace` 的核心順序是：

1. Observe.
2. Reason/classify.
3. Answer.

我們把這個順序放進 `caption_template.md`：

```text
Observe first, classify the visual pattern second, then report uncertainty.
Do not expose chain-of-thought. Provide concise visual evidence only.
```

這樣可以避免模型一開始就跳到 clinical conclusion。

### 1.1.5 從 VLM-AutoDrive 借到的 prompt pattern

我們借到的 pattern：

- `dimension coverage`: 強制掃過固定視覺維度。
- `mandatory checks`: 對重要 feature 設定必答問題。
- `evidence-bound reasoning`: 任何推論都要對應 visual evidence。
- `hidden/internal checklist`: 讓模型按順序觀察，但 final output 不放長篇推理。
- `structured output`: 用 JSON 讓後續 pipeline 能做 validation 和 scoring。

---

## 1.2 MM-Skin Captioning & VQA File 的邏輯

MM-Skin 的資料不是直接拿來當 wound ground truth。它的價值在於：

- dermatology image caption 的語言風格。
- medical visual vocabulary。
- clinical VQA 的問句設計。

## 1.2.1 Caption CSV 的邏輯

`reference/mmskinCaption.csv` 的資料形態大致是：

```text
image, caption, modality, sex, age, cleaned_caption
```

它的 caption 很像 textbook-style image description，常見特徵：

- 會描述 lesion type。
- 會描述 body site。
- 會描述 color / shape / texture / distribution。
- 會使用 dermatology terms。
- 有時會包含 diagnosis name、histology、病程背景。

對本專案有用的是 vocabulary，不是疾病 label。

可抽出的詞彙包括：

- erythema
- edema
- blister / bullae
- ulceration
- erosion
- necrosis
- eschar
- exudate
- scar / fibrosis
- hemorrhage
- discoloration
- scaling
- crusting
- smooth
- shiny
- rough
- infiltrated
- plaque
- nodule
- maceration

這些詞被放進 `caption_template.md` 的 `CLINICAL VOCABULARY BANK`，讓 local VLM 的輸出比較接近 clinical annotation。

### 限制

MM-Skin caption 不能直接當 wound dataset 的 ground truth，原因：

- 它主要是 dermatology disease，不是 acute traumatic wound。
- 很多 caption 包含 diagnosis，不是純 visual observation。
- 部分文字有 OCR noise。
- 它的 label space 不等於本專案的 wound categories。

所以使用方式是：

```text
拿 vocabulary，不拿 label。
拿描述風格，不拿醫療結論。
```

## 1.2.2 VQA CSV 的邏輯

`reference/mmskinVQA.csv` 的資料形態是：

```text
image, question, answer
```

它的問題常聚焦在：

- main focus
- texture
- color
- shape
- size
- location
- inflammation
- redness / swelling
- evidence
- feature present or absent
- difference between lesion patterns

常見問句型態：

```text
Can you describe the texture of the skin?
Is there any evidence of inflammation?
What is the color of the lesion?
What is the main focus of this image?
Can you describe the size or shape?
Is there any exudate present?
Are there any signs of necrosis?
```

我對它 prompt design 的推測：

- 它不是單純 caption expansion。
- 它把 dermatology image semantics 拆成固定 observation questions。
- 透過 QA 讓模型學會針對單一 visual feature 回答。
- 問題不是全部 diagnosis-oriented，而是大量 feature-oriented。

這直接啟發我們的 `Auxiliary VQA Prompt`。

## 1.2.3 MM-Skin 對本專案的定位

MM-Skin 可以當：

- vocabulary bank
- question template source
- positive/negative feature wording source
- dermatology caption style reference

MM-Skin 不應該當：

- wound category label truth
- burn/cut/laceration 的 clinical standard
- final evaluation reference
- 專家標註

---
## 2. Current Prompt Design

目前 prompt 有兩個 mode：

- `simple`
- `full`

這兩個 mode 是為了做 ablation，不是二選一。

`simple` 問：

```text
單段 prompt 可以做到什麼？
```

`full` 問：

```text
加上 dimension coverage + auxiliary VQA + self-check 後，結果是否更穩？
```

---

## 2.1 Simple Prompt

檔案：

```text
prompts/simple_caption_template.md
```

定位：

```text
single-stage baseline
```

它只做一次 VLM call。

## 2.1.1 Simple Prompt 的內容

Simple prompt 要求模型：

- 描述 visible wound image。
- 使用 target category guidance。
- 輸出 JSON。
- 不要 markdown。
- 不要 diagnosis。
- 不要 invent findings。
- feature absent 時寫 `not observed`。
- uncertain 時寫 `uncertain`。
- 沒有 ruler / calibration card 不估尺寸。

輸出 shape 與 `wound_schema.json` 對齊：

- `image_observation`
- `wound_features`
- `category_specific_check`
- `uncertainty`
- `safety_scope`

## 2.1.2 Simple Prompt 的用途

Simple prompt 用途：

- 低成本 baseline。
- 快速 smoke test。
- local/SaaS 都能跑。
- 觀察模型在最少 prompt engineering 下的能力。
- 和 full prompt 做 2x2 ablation。

## 2.1.3 Simple Prompt 的風險

Simple prompt 的風險：

- 容易漏掉 wound feature。
- 對 `not observed` 的要求較弱。
- 沒有額外 VQA 檢查。
- 沒有 self-check。
- hallucination control 完全依賴單段 rules。
- category prompt 提到的 feature 可能誘導模型寫出圖上沒有的東西。

---

## 2.2 Full Prompt

Full prompt 是 multi-stage prompt engineering version。

相關檔案：

```text
prompts/caption_template.md
prompts/auxiliary_vqa_template.md
prompts/self_check_template.md
```

Full mode 每張圖約 3 次 model call：

1. Caption Prompt
2. Auxiliary VQA Prompt
3. Self-Check Prompt

如果 pipeline 同時跑：

```text
--modes simple,full
```

那每張圖總共約 4 次 model call：

- simple: 1
- full: 3

## 2.2.1 Stage 1: Caption Prompt

檔案：

```text
prompts/caption_template.md
```

目的：

```text
產生主要 wound image caption JSON。
```

它比 simple prompt 多了明確的 dimension coverage。

強制覆蓋：

- Location/body site
- Wound type cue
- Shape and spatial pattern
- Edge or margin quality
- Wound bed or visible tissue state
- Color distribution
- Texture
- Fluid, exudate, crust, pus, active bleeding
- Periwound skin
- Depth cues, only if visually supported
- Foreign material or debris
- Necrosis or eschar
- Measurement tool visibility
- Uncertainty and features not observed

這對應 VLM-AutoDrive 的 `Understanding VQA` dimension coverage。

它也加入 `CLINICAL VOCABULARY BANK`，對應 MM-Skin caption 的 dermatology terms。

## 2.2.2 Stage 2: Auxiliary VQA Prompt

檔案：

```text
prompts/auxiliary_vqa_template.md
```

目的：

```text
用固定 clinical observation questions 補 caption coverage。
```

固定 10 題：

1. What is the main visible lesion or injury?
2. Is the epidermis visibly broken?
3. What is the dominant color pattern?
4. Are the wound edges clean, jagged, raised, or unclear?
5. Is there visible bleeding, exudate, crust, or pus?
6. Is there blistering or bullae?
7. Is necrosis or eschar visible?
8. Is the surrounding skin erythematous, edematous, bruised, discolored, or normal?
9. Is there a ruler, calibration card, or measurement object?
10. Which expected category features are not observed?

這對應 MM-Skin VQA 的 prompt design：把 clinical image observation 拆成 question-answer units。

Auxiliary VQA 的 output 不是 final report，而是輔助檢查資料。

用途：

- 補 caption 漏掉的 feature。
- 檢查 category prompt 是否誘導 hallucination。
- 幫 self-check 找 contradiction。
- 未來可以做 feature-level scoring。

## 2.2.3 Stage 3: Self-Check Prompt

檔案：

```text
prompts/self_check_template.md
```

目的：

```text
cheap validation。
```

Self-check 讀：

- candidate caption JSON
- auxiliary VQA JSON
- category guidance

它檢查：

- JSON 是否完整。
- 是否混入 markdown/table/prose。
- 是否有 diagnosis wording。
- 是否沒有 measurement tool 卻估尺寸。
- 是否提到 VQA 說 `not observed` 或 `uncertain` 的 feature。
- clinical inference 是否缺 visual evidence。
- uncertainty / not-observed 是否有被表示。

注意：

```text
Self-check 不是專家審查。
Self-check 只是 prompt compliance check。
```

## 2.2.4 Full Prompt 的優點

Full prompt 優點：

- feature coverage 更高。
- `not observed` 使用更穩。
- 對 measurement hallucination 控制較好。
- 對 category-induced hallucination 有額外檢查。
- output 更適合後續 validation / scoring。
- 可以更清楚分離 visual observation 與 clinical inference。

## 2.2.5 Full Prompt 的缺點

Full prompt 缺點：

- 慢。
- local GPU inference 更耗時。
- SaaS API 成本更高。
- self-check 還是模型輸出，不是 ground truth。
- 如果 VLM 本身看不懂圖，prompt 只能改善 coverage 和格式，不能創造視覺能力。

---

## 3. Category Prompt Design

目前每個 category prompt 都拆成四段：

```text
POSITIVE CUES
NEGATIVE REPORTING
CONFUSABLE CATEGORIES
FORBIDDEN INFERENCE
```

這是為了解決早期 prompt 的問題：模型看到 category label 後，容易把典型特徵硬寫出來。

## 3.1 POSITIVE CUES

這段列出該類 wound 的 expected visual evidence。

例如 burn：

- erythematous burned skin
- blistering / bullae
- peeled epidermis
- moist wound bed
- waxy surface
- leathery eschar

這些是 checklist，不是必填答案。

看到才寫進：

```text
observed_supporting_features
```

## 3.2 NEGATIVE REPORTING

這段要求模型：

```text
沒看到就寫 not observed。
```

目的：

- 降低 hallucination。
- 避免 category prompt 誘導。
- 讓 output 可以被 not-observed usage rate 評估。

例如 burn prompt 要求：

- 沒看到 blister/bullae，要列入 `expected_but_not_observed`。
- 沒看到 eschar/necrosis，要寫 `necrosis_eschar: not observed`。
- 看不出 depth，要放進 `cannot_determine`。

## 3.3 CONFUSABLE CATEGORIES

這段列出容易混淆的類別。

最重要的混淆組：

- cut / incision
- laceration
- stab wound / puncture wound
- abrasion

區分邏輯：

- cut/incision: clean linear edge, sharp margin
- laceration: jagged torn margin, tissue flap, tissue bridging
- stab/puncture: small entry point, depth cue greater than width
- abrasion: superficial scraped surface, friction pattern
- bruise: intact skin discoloration
- burn: blistering, thermal surface pattern, waxy/leathery/moist texture

## 3.4 FORBIDDEN INFERENCE

這段限制模型不要做超出 visual annotation 的事。

禁止：

- diagnosis
- treatment recommendation
- infection conclusion
- abuse/coagulopathy/systemic disease 推測
- 無尺估尺寸
- 無 visual evidence 判 depth
- 推測造成傷口的物體、機制、時間
- 推測內部組織損傷

這對醫療影像很重要，因為本專案輸出不是 medical diagnosis，而是 wound image-captioning baseline。

---

## 4. Simple vs Full 詳細比較

| 面向 | Simple | Full |
| :--- | :--- | :--- |
| 主要用途 | single-stage baseline | prompt engineering version |
| model call 次數 | 1 | 約 3 |
| pipeline output | caption JSON | caption JSON + auxiliary VQA JSON + self-check JSON |
| dimension coverage | 中等 | 高 |
| negative reporting | 較弱 | 較強 |
| hallucination control | 靠單段 rules | caption + VQA + self-check |
| 成本 | 低 | 高 |
| 速度 | 快 | 慢 |
| 適合階段 | smoke、baseline、快速比較 | formal、prompt performance 檢查 |
| 主要風險 | 漏特徵、category 誘導 | 成本高、self-check 仍非專家審查 |

Simple 的意義：

```text
測模型在最少 prompt engineering 下可以做到什麼。
```

Full 的意義：

```text
測 domain transfer + dimension coverage + VQA check + self-check 是否真的改善結果。
```

所以 simple 和 full 必須都保留，否則無法證明 prompt engineering 有效。

---

## 5. Evaluation Connection

Prompt 設計會進入 2x2 ablation：

| Provider | Simple | Full |
| :--- | :--- | :--- |
| local | local + simple | local + full |
| SaaS | SaaS + simple | SaaS + full |

可以回答：

- SaaS 上 full 是否比 simple 穩定？
- local 上 full 是否比 simple 穩定？
- local full 是否更接近 SaaS full reference？
- prompt engineering 是否改善 schema validity、feature coverage、negative reporting？

## 5.1 Smoke Scope

`smoke` 是抽樣子集，預設 10%。

用途：

- 快速檢查 pipeline。
- 快速看 SaaS reference 品質。
- 快速比較 simple/full 是否有明顯差異。
- 避免一開始就花大量 GPU/API 成本。

Smoke 不適合當 final conclusion。

## 5.2 Formal Scope

`formal` 是 100% supported wound dataset。

用途：

- 正式 2x2 ablation。
- 正式分數表。
- 報告用結果。

## 5.3 Score 解讀

Behavior metrics：

- `schema_valid_rate`: JSON/schema 穩定性。
- `required_field_completion_rate`: 必填 visual fields 完整度。
- `not_observed_usage_rate`: negative reporting 是否生效。
- `no_ruler_size_compliance_rate`: 沒有 ruler/calibration card 時是否避免尺寸 hallucination。
- `safety_scope_compliance_rate`: 是否維持 visual description only。

Text similarity metrics：

- BLEU-4
- ROUGE-L
- METEOR-lite
- CIDEr-lite

這些比較 local output 和 SaaS reference 的文字相似度。

注意：這不是 clinical correctness 的最終證明。正式醫療有效性仍需要 manual review 或 domain expert check。

---

## 6. 結論

目前 prompt design 的核心是：

```text
用 VLM-AutoDrive 的 dimension/reasoning structure，
加上 MM-Skin 的 dermatology vocabulary/VQA 問法，
設計成 simple baseline 與 full multi-stage prompt，
最後用同一套 JSON schema 和 evaluation pipeline 比較。
```

這樣可以讓 prompt engineering 不只是主觀「看起來比較好」，而是能用 schema validity、feature coverage、negative reporting、no-ruler compliance、local-vs-SaaS similarity 做量化比較。
