# BERTScore 計算指南

## 1. 目前用途

本專案使用 BERTScore 作為 wound caption 的探索性語意相似度指標。它不直接衡量醫療正確性、visual grounding 或診斷安全性。

實作位於：

```text
scripts/bertscore_rescore.py
```

一般的 `pipeline.py score` 只計算 BLEU-4、ROUGE-L、METEOR-lite 與 CIDEr-lite，不會計算 BERTScore。

## 2. 正式 comparison

目前只支援兩組 comparison：

| ID | Reference | Candidate | 語言 | Upstream 預設模型 |
|---|---|---|---|---|
| `AC` | SaaS/Gemini 英文 caption | Local 英文 caption | `en` | `roberta-large` |
| `BD` | SaaS caption 的中文翻譯 | Local caption 的中文翻譯 | `zh` | `bert-base-chinese` |

`AC` 與 `BD` 使用不同語言模型與 baseline，必須分開報告，不可平均成單一分數。

目前實作不包含跨語言的 `AB`、`CD`、`AD` comparison。

## 3. 計分 scope

每份 report 僅包含一個 comparison 與一個 scope。

### `visual_summary_only`

只使用：

```text
caption.image_observation.visual_summary
```

用途是評估自由形式的視覺描述，降低重複 schema 欄位對分數的影響。

### `full_caption_fields`

串接以下欄位：

```text
caption.image_observation.visual_summary
caption.wound_features.shape_pattern
caption.wound_features.edges_margins
caption.wound_features.wound_bed
caption.wound_features.color
caption.wound_features.texture
caption.wound_features.fluid_exudate_bleeding
caption.wound_features.periwound_skin
caption.category_specific_check.observed_supporting_features
```

此 scope 評估選定 structured wound-description 欄位的語意相似度，不代表完整輸出 JSON。

## 4. Upstream BERTScore 方法

程式直接呼叫官方 `bert_score.score()`。BERTScore 使用 contextual token embeddings 與 greedy cosine-similarity matching，輸出 Precision、Recall、F1。

- Precision：candidate token 是否能在 reference 中找到相近語意。
- Recall：reference 的內容是否被 candidate 涵蓋。
- F1：Precision 與 Recall 的 harmonic mean。

AC 傳入 `lang=en`，BD 傳入 `lang=zh`，且不覆寫 `model_type` 或 `num_layers`。因此模型與 tuned layer 由 upstream package 依語言選擇。

## 5. 官方 baseline rescaling

正式計算固定啟用：

```python
rescale_with_baseline=True
```

Upstream package 對 Precision、Recall、F1 分別套用對應 model/language/layer baseline：

```text
rescaled_score = (raw_score - baseline) / (1 - baseline)
```

本專案不再套用 `(raw + 1) / 2`、`[0, 1]` clipping 或其他後處理轉換。

報告中的 `bertscore_precision`、`bertscore_recall`、`bertscore_f1` 均為官方 baseline-rescaled output。Rescaled score 不保證一定落在 `[0, 1]`。

## 6. Baseline 驗證

模型推論前，程式會確認已安裝的 `bert-score` package 內含 upstream 選定語言與模型的 baseline 檔案。

預期組合：

```text
AC: rescale_baseline/en/roberta-large.tsv
BD: rescale_baseline/zh/bert-base-chinese.tsv
```

若 baseline 不存在，command 會直接終止，避免把未 rescale 的值誤標為 rescaled score。

## 7. 可重現性

程式呼叫：

```python
bert_score.score(..., return_hash=True)
```

每個完成計分的 scope 都記錄：

```text
lang
model_type
num_layers
baseline_path
bert_score_version
transformers_version
torch_version
bertscore_hash
score_representation
```

`bertscore_hash` 是比較設定時的主要識別資訊。Hash、tokenizer、IDF、model 或 score representation 不同的報告，不應視為可直接比較。

`requirements.txt` 已將 `bert-score` 固定為 `0.3.13`。

## 8. IDF

IDF weighting 為選用功能，預設關閉：

```text
--idf
```

啟用後，IDF 由該次 command 的 reference corpus 計算。輸入集合或 `--limit` 改變時，IDF 與結果也可能改變。Reference 數量太少時 IDF 不穩定，因此正式報告必須記錄是否啟用。

## 9. Model 檔案

安裝 `bert-score` 會安裝計分程式與 package 內建 baseline TSV，但不一定已下載 Hugging Face model weights。

第一次執行 AC 或 BD 時，Transformers 會下載對應 tokenizer/model；若 Hugging Face cache 已存在則直接使用，不會重複下載。

預期模型：

```text
AC: roberta-large
BD: bert-base-chinese
```

## 10. 配對與彙總

JSON 目前依 filename stem 配對，報告會記錄：

```text
matched_image_count
unmatched_image_count
unmatched_images
scored_image_count
empty_text_pair_count
```

Candidate 或 reference 文字為空的 pair 不參與計分。結果分為三層：

- 全體平均；
- wound category 平均；
- per-image evidence，包含實際 candidate/reference text。

每個來源資料夾內的 filename stem 必須唯一，否則可能覆寫或錯誤配對。

## 11. 結果解讀

BERTScore 應解讀為語意重疊訊號：

- 數值較高代表在指定 model 下 embedding similarity 較高。
- 不能證明 caption 的醫療內容正確。
- 不能單獨偵測缺乏視覺證據的敘述。
- 沒有通用的 acceptable threshold。
- AC 是主要英文 caption-quality comparison。
- BD 是獨立的中文翻譯審閱比較，分數包含翻譯造成的影響。

應搭配 schema validity、paired-image coverage、per-image evidence、VQA 結果與人工審閱一起解讀。

## 12. Legacy report

Report schema version 2 以前的報告使用專案自訂轉換：

```text
normalized = clamp((raw + 1) / 2, 0, 1)
```

Legacy 數值無法和目前官方 baseline-rescaled 數值直接比較。正式結果應使用目前 script 重新計算。
