# BERTScore 解讀與門檻政策

## 目前結論

本專案目前不定義通用的 BERTScore pass threshold。

BERTScore 衡量 contextual semantic similarity，適合用於比較或排序生成 caption；它不直接衡量醫療正確性、visual grounding 或臨床可接受性。

在沒有專案人工校準結果前，`0.70-0.75` 等數值不可直接描述為 good 或 acceptable。

## Score representation

目前 report 使用官方 `bert-score` baseline-rescaled output：

```text
rescaled_score = (raw_score - baseline) / (1 - baseline)
```

Baseline 由 upstream package 依 language、model、layer 選擇。

本專案已停止使用：

```text
normalized = clamp((raw + 1) / 2, 0, 1)
```

因此：

- Legacy project-normalized report 不可與目前 report 直接比較。
- 官方 rescaled value 不保證一定落在 `[0, 1]`。
- 引用分數時必須同時提供 BERTScore hash。

## 正式 comparison 政策

| Comparison | 語言 | Upstream 預設模型 | 用途 |
|---|---|---|---|
| `AC` | English (`en`) | `roberta-large` | 主要英文 caption-quality comparison |
| `BD` | Chinese (`zh`) | `bert-base-chinese` | 獨立的中文翻譯人工審閱比較 |

AC 與 BD 使用不同語言、模型、baseline 與文字產生路徑，不得合併為單一分數。

目前 scoring pipeline 不包含跨語言 AB/CD/AD comparison。

## 為何不使用通用 cutoff

BERTScore 原始研究主要驗證 metric 與 human judgment 的相關性，沒有建立 wound-caption pair-level cutoff。

本專案與常見 benchmark 還有以下差異：

- 每個 generated caption 通常只有一個 SaaS reference。
- SaaS reference 不是 medical gold annotation。
- Wound caption 含有重複的 structured fields。
- BD 另外包含翻譯誤差。
- 各 run 的 category distribution 與 caption length 可能不同。

因此，單一絕對分數無法證明 caption 已達可接受標準。

## 報告分數時必須附帶的資訊

每個 BERTScore 結果應同時提供：

```text
comparison（AC 或 BD）
scope（visual_summary_only 或 full_caption_fields）
language
model
layer
IDF setting
official baseline-rescaled status
BERTScore hash
matched/scored image counts
```

若 report 的 BERTScore hash 或 score representation 不同，除非差異本身就是實驗變因，否則不可直接比較。

## 建議閱讀順序

1. 確認 report 使用 schema version 2 與 `official_baseline_rescaled`。
2. 確認 BERTScore hash、language 與 model。
3. 檢查 matched、unmatched、scored、empty-text counts。
4. 先看 AC `visual_summary_only`，作為主要 caption semantic signal。
5. 分開查看 AC `full_caption_fields` 的 structured-field 表現。
6. 檢查 category variation 與 per-image evidence。
7. 再獨立查看 BD，了解中文翻譯審閱視角。
8. 最後搭配 schema validity、VQA evidence 與人工審閱。

## 未來校準方式

如果未來需要專案專屬 threshold，應建立人工標註 calibration set，例如：

```text
acceptable
needs_revision
unsafe_or_unsupported
```

接著分析官方 rescaled AC score 在各人工 label 下的分布，依需要的 precision/recall tradeoff 選擇門檻，並用獨立 held-out set 驗證。

完成校準前，BERTScore 應描述為探索性 semantic-similarity metric，而不是 pass/fail criterion。

## Legacy 處理

舊 report 若同時包含 `raw_bertscore_*` 與 project-normalized `bertscore_*`，應標記為 legacy，不得再次轉換或混入目前 aggregate table。應直接使用目前 scorer 對原始 caption 重新計算。
