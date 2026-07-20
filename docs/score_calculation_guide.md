# BERTScore Calculation Guide

本文件只說明目前專案使用的 BERTScore 計算方式。舊的 `BLEU-4`、`ROUGE-L`、`METEOR-lite`、`CIDEr-lite` 不再放在本 guide，避免和 620 階段的 semantic score 解讀混在一起。

主要 script：

```text
scripts/bertscore_rescore.py
```

主要參考：

```text
docs/bertScore.pdf
Tianyi Zhang et al., BERTScore: Evaluating Text Generation with BERT, ICLR 2020.
```

## 1. BERTScore 在本專案的用途

本專案用 BERTScore 回答：

```text
candidate caption 和 reference caption 在語意上有多接近？
```

它不能回答：

```text
影像描述是否醫學正確？
傷口分類是否正確？
caption 是否足夠可用於 clinical decision？
```

因此 BERTScore 必須搭配：

- per-image evidence text。
- `visual_summary_only` vs `full_caption_fields` delta。
- schema validity。
- VQA-style check。
- human review。

## 2. A/B/C/D Data Source

620 之後，BERTScore 的四個資料來源必須由 command 明確指定。script 不會自動搜尋其他 run folder。

| ID | Meaning | Command Argument |
|---|---|---|
| `A` | reference English caption | `--reference-dir` |
| `B` | translated reference caption | `--reference-translation-dir` |
| `C` | candidate English caption | `--candidate-dir` |
| `D` | translated candidate caption | `--candidate-translation-dir` |

原因：

```text
如果某個 run 只有 local/simple output，script 不應該自動拿其他 run 的 translation 補 B/D。
跨 run source 只能在 command 中明確指定，否則結果不可追蹤。
```

範例 command：

```bash
python scripts/bertscore_rescore.py \
  --run-dir runs/20260525_gemma4_26b \
  --reference-dir runs/saas_simple_baseline \
  --reference-translation-dir runs/510_smoke_v1/output_translation/saas/simple \
  --candidate-dir runs/20260525_gemma4_26b/outputs/local/simple \
  --candidate-translation-dir runs/20260525_gemma4_26b/output_translation/local/simple \
  --provider local \
  --mode simple \
  --score-type grouped \
  --output-dir runs/bertscore_reports \
  --report-id 20260525_gemma4_26b_local_simple_vs_saas_simple \
  --lang zh \
  --model-type bert-base-multilingual-cased
```

## 3. Main Comparison Groups

目前主結果只看：

| Comparison | Reference | Candidate | Meaning |
|---|---|---|---|
| `AC` | `A` | `C` | English caption quality：local English vs SaaS/Gemini English reference |
| `BD` | `B` | `D` | Chinese human-review view：translated local vs translated reference |

Diagnostic comparison 只有在加上 `--include-diagnostic` 時才跑：

| Comparison | Meaning |
|---|---|
| `AB` | reference translation faithfulness |
| `CD` | candidate translation faithfulness |
| `AD` | translated candidate vs English reference |

重點：

```text
AC 和 BD 是兩個不同視角，不應平均成單一 final score。
```

## 4. Scoring Scope

每次 command 由 `--scopes` 指定要算哪個 scope。

建議平常分開跑：

```bash
--scopes visual_summary_only
--scopes full_caption_fields
```

若需要在同一份 report 內計算 `delta_f1`，才使用：

```bash
--scopes visual_summary_only,full_caption_fields
```

### 4.1 visual_summary_only

只抽：

```text
caption.image_observation.visual_summary
```

用途：

```text
檢查模型自由描述的核心語意是否接近 reference。
```

### 4.2 full_caption_fields

抽以下欄位並串成一段文字：

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

用途：

```text
檢查完整 structured wound caption 的語意接近度。
```

風險：

```text
fixed short fields，例如 not observed / unknown / red / irregular，可能拉高或拉低整體分數。
```

## 5. Raw BERTScore P / R / F1 是什麼意思？

這是最容易誤解的地方。

在 raw BERTScore 裡：

```text
precision / recall 不是 classification precision / recall。
```

它們不是：

```text
TP / (TP + FP)
TP / (TP + FN)
```

它們是：

```text
directional token-embedding similarity averages
```

原論文 Section 3, PDF p.4 說明：BERTScore 先把 reference 和 candidate token 轉成 contextual embeddings，再計算 token 之間的 cosine similarity。每個 token 會在另一句中找最相似的 token。

### 5.1 Raw Precision

`P_BERT` 的方向是：

```text
candidate token -> reference token
```

意思：

```text
candidate 產生的每個 token，在 reference 裡是否能找到語意相近的支持？
```

在本專案可解讀為：

```text
precision 較低時，candidate 可能加入 reference 沒有支持的內容。
```

但注意：

```text
這仍然只是 embedding similarity，不是 hallucination detector。
```

### 5.2 Raw Recall

`R_BERT` 的方向是：

```text
reference token -> candidate token
```

意思：

```text
reference 中的重要 token，在 candidate 裡是否能找到語意相近的覆蓋？
```

在本專案可解讀為：

```text
recall 較低時，candidate 可能漏掉 reference 描述的內容。
```

### 5.3 Raw F1

`F_BERT` 是 raw precision 和 raw recall 的 harmonic mean：

```text
F_BERT = 2 * P_BERT * R_BERT / (P_BERT + R_BERT)
```

意思：

```text
同時考慮 candidate 是否被 reference 支持，以及 reference 是否被 candidate 覆蓋。
```

### 5.4 raw score 的尺度問題

原論文 Section 3, Baseline Rescaling, PDF p.5 說明：raw BERTScore 使用 cosine similarity，理論範圍可在 `-1` 到 `1`，但實務上常落在較窄範圍。

所以 raw `0.70` 不等於：

```text
70% correct
70% token overlap
70% medical correctness
```

它比較像：

```text
平均 token-level contextual embedding similarity 的 directional summary。
```

這也是為什麼原論文提出 rescaling：不是改變 ranking ability，而是增加 readability。

### 5.5 本專案輸出欄位：normalized vs raw

2026-07-03 後，`scripts/bertscore_rescore.py` 會在 BERTScore library 回傳 raw P/R/F1 後，再套一層固定標準化：

```text
normalized = clamp((raw + 1) / 2, 0, 1)
```

因此 report 裡主要欄位是 normalized score：

```text
bertscore_precision
bertscore_recall
bertscore_f1
```

原始 BERTScore library output 會保留在：

```text
raw_bertscore_precision
raw_bertscore_recall
raw_bertscore_f1
```

意思：

```text
bertscore_*     = 方便閱讀的 0-1 標準化版本
raw_bertscore_* = 原始 cosine-like BERTScore output
```

注意：

```text
normalized 0.85 不等於 85% correct。
它只是把 raw cosine-like 分數從 -1~1 線性映射到 0~1。
```

## 6. Rescale 是什麼？

原論文 Section 3, Baseline Rescaling, PDF p.5 說明：

- raw BERTScore 的實際值常落在窄範圍，不易讀。
- 作者用 Common Crawl monolingual data 建立 random candidate-reference pairs。
- 這些 random pairs 語意重疊很低。
- 將它們的平均 BERTScore 當作 empirical lower bound `b`。
- 再線性 rescale：

```text
R̂_BERT = (R_BERT - b) / (1 - b)
```

同樣方法用於 `P̂_BERT` 和 `F̂_BERT`。

重要解讀：

```text
rescale 只是增加 score readability。
原論文說它不影響 ranking ability / human correlation。
```

本專案目前預設：

```text
rescale_with_baseline = False
```

所以目前不使用原論文 baseline rescaling。專案 report 的主 P/R/F1 是本專案後處理的 0-1 normalized score，raw 值另存於 `raw_bertscore_*`。

## 7. Delta F1

只有當同一份 report 同時包含 `visual_summary_only` 和 `full_caption_fields` 時，才會算：

```text
delta_f1 = full_caption_fields_f1 - visual_summary_only_f1
```

解讀：

| Flag | Rule | Meaning |
|---|---|---|
| `none` | `abs(delta_f1) < 0.05` | structured fields 沒有明顯影響 |
| `noticeable` | `abs(delta_f1) >= 0.05` | structured fields 對分數有可見影響 |
| `strong` | `abs(delta_f1) >= 0.10` | structured fields 可能明顯拉高或拉低 |

這個 delta 是 620 階段加上的 calibration tool，用來避免 `full_caption_fields` 被固定欄位誤導。

如果你用兩個獨立 command 分開跑 scope，兩份 report 內不會自動有 `delta_f1`。這時候兩個 scope 的比較要在外部 summary 或人工 review 中對照。

## 8. Matching / Empty Text / Summary 怎麼算

### 8.1 Matching

script 以 JSON stem 配對：

```text
matched_images = sorted(set(reference_files) & set(candidate_files))
unmatched_images = sorted(set(reference_files) ^ set(candidate_files))
```

BERTScore 只在 matched images 上計算。unmatched image 會記錄，但不直接拉低 P/R/F1。

### 8.2 Empty Text Filtering

matched image 之後，script 依 scope 抽文字：

```text
visual_summary_only -> visual_summary_text()
full_caption_fields -> caption_text()
```

如果 reference 或 candidate 文字為空，該 pair 不進入 BERTScore batch。

因此：

```text
matched_image_count >= scored_image_count
```

### 8.3 Per-Image Score

每張有效圖片會得到：

```text
bertscore_precision
bertscore_recall
bertscore_f1
reference_text
candidate_text
```

位置：

```text
summary.<comparison_id>.scopes.<scope>.by_image[]
```

Markdown report 也會列出 per-image evidence，方便人工檢查。

### 8.4 Category Score

Category table 是在同一個 comparison + scope 內，依 category 分組後平均。

公式：

```text
category_precision = mean(image_precision for images in this category)
category_recall    = mean(image_recall    for images in this category)
category_f1        = mean(image_f1        for images in this category)
```

範例：

```text
AC / full_caption_fields / bruises
```

只會平均 `AC` + `full_caption_fields` 中 category 是 `bruises` 的圖片，不會混入 `BD` 或 `visual_summary_only`。

### 8.5 Summary Score

Summary score 是同一個 comparison + scope 內，對所有 scored images 平均。

公式：

```text
summary_precision = mean(image_precision for all scored images in this comparison + scope)
summary_recall    = mean(image_recall    for all scored images in this comparison + scope)
summary_f1        = mean(image_f1        for all scored images in this comparison + scope)
```

注意：

```text
summary_f1 不是 category_f1 的平均。
summary_f1 是所有 scored image 的 image-level F1 平均。
```

## 9. 原論文 COCO Image Captioning 實驗

原論文 Section 4, PDF p.6 的 image-captioning 實驗使用 COCO 2015 Captioning Challenge。

### 9.1 Dataset / System Setup

paper 說明：

- 使用 12 個 COCO 2015 Captioning Challenge submission entries。
- 每個 participating system 會替 COCO validation set 的每張 image 產生一個 caption。
- 每張 image 約有 5 個 human reference captions。

paper 位置：

```text
Section 4 Experimental Setup - Image Captioning, PDF p.6
```

### 9.2 Human Judgment

原論文跟隨 Cui et al. (2018)，使用兩個 system-level human judgment metrics：

| Human Metric | Meaning |
|---|---|
| `M1` | captions 被評為 better or equal to human captions 的比例 |
| `M2` | captions 被評為 indistinguishable from human captions 的比例 |

這代表 Table 5 不是在看單張 image pair 的品質，而是看每個 system 的整體 human-rated quality。

### 9.3 Multiple References 怎麼算

COCO 每張圖有多個 reference captions。原論文 Section 4, PDF p.6 說：

```text
compute BERTScore with multiple references by scoring the candidate with each available reference and returning the highest score
```

也就是：

```text
image_score = max(BERTScore(candidate, reference_i) for reference_i in references)
```

這和本專案目前不同。本專案目前通常是：

```text
one candidate JSON vs one reference JSON
```

所以本專案 score 可能比 COCO-style multi-reference 更嚴格，因為沒有五個 human references 可取 max。

### 9.4 Table 5 到底報什麼

原論文 Table 5, PDF p.8 報的是：

```text
Pearson correlation between metric scores and system-level human metrics M1/M2
```

不是：

```text
caption pair raw BERTScore
```

Table 5 部分數字：

| Metric | M1 | M2 |
|---|---:|---:|
| BLEU | -0.019 | -0.005 |
| METEOR | 0.606 | 0.594 |
| ROUGE-L | 0.090 | 0.096 |
| CIDEr | 0.438 | 0.440 |
| SPICE | 0.759 | 0.750 |
| LEIC | 0.939 | 0.949 |
| RBERT | 0.888 | 0.863 |
| FBERT | 0.322 | 0.350 |
| RBERT(idf) | 0.917 | 0.889 |

paper 位置：

```text
Table 5, PDF p.8
```

重點：

```text
Table 5 的 0.888 / 0.917 是 correlation，不是 BERTScore 本身。
不能拿 Table 5 直接說 pair-level BERTScore 0.70-0.75 是否 good。
```

### 9.5 COCO 結果對本專案能支持什麼

能支持：

```text
BERTScore 類方法在 image captioning system-level evaluation 上可與 human judgment 有高 correlation。
```

不能支持：

```text
0.70-0.75 是 wound captioning 的通用合格門檻。
```

## 10. 0.70-0.75 如何解讀

若本專案 report 使用 normalized BERTScore，且 `rescale_with_baseline = False`：

```text
0.70-0.75 = normalized semantic overlap signal，但不能直接說足夠好。
```

更精確：

```text
normalized P/R/F1 是 raw token embedding similarity summary 經 (raw + 1) / 2 後的閱讀版，不是百分比正確率。
```

建議寫法：

```text
BERTScore 0.70-0.75 is treated as an exploratory semantic-similarity signal.
The original BERTScore paper validates correlation with human judgments, not a universal absolute threshold.
Therefore, this score range must be calibrated with per-image evidence, schema validity, VQA-style checks, and human review.
```

## 11. Report 解讀 Checklist

看 BERTScore report 時，照這個順序：

1. 先確認 A/B/C/D path 是否正確。
2. 看 AC，不要先看 BD。
3. 看 `visual_summary_only`，確認自由描述本身是否接近。
4. 看 `full_caption_fields`，確認 structured fields 是否改變分數。
5. 看 `delta_f1`：
   - full 比 visual 高很多：可能被固定欄位拉高。
   - full 比 visual 低很多：structured fields 可能揭露更多差異。
6. 看 per-image evidence text，不要只看 summary。
7. 再看 BD，確認中文 review 視角是否和 AC 一致。
8. 最後才做 human conclusion。

## 12. Paper Evidence

| Claim | Evidence |
|---|---|
| BERTScore 是 contextual embedding token similarity | Section 3, PDF p.4 |
| Raw P/R/F1 是 directional max cosine similarity averages | Section 3, PDF p.4 |
| Baseline rescaling 只為 readability，不改 ranking/correlation | Section 3 Baseline Rescaling, PDF p.5 |
| 作者建議 F1 作為穩定 measure | Section 5 Results, PDF p.7 |
| COCO image-captioning 使用 12 systems、COCO validation、約 5 references/image | Section 4 Image Captioning, PDF p.6 |
| COCO multiple references 是 candidate 對每個 reference scoring 後取最高 | Section 4 Image Captioning, PDF p.6 |
| COCO Table 5 報 Pearson correlation，不是 pair-level BERTScore | Table 5, PDF p.8 |
