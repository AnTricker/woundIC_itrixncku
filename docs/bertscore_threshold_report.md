# BERTScore 0.70-0.75 是否足夠好？

本文回應問題：

```text
BERTScore 約 0.70-0.75 是否足夠好，需要文獻依據支持。
不能只說分數看起來可以，需確認 BERTScore 原始論文或相關資料集中，類似任務的分數範圍如何被解讀。
```

使用資料來源：

```text
docs/bertScore.pdf
Tianyi Zhang et al., BERTScore: Evaluating Text Generation with BERT, ICLR 2020.
```

## 1. 結論

`BERTScore = 0.70-0.75` 不能直接說「足夠好」。

比較嚴謹的說法應該是：

```text
在目前 wound caption / translation review 任務中，BERTScore 0.70-0.75 可視為「有中等語意重疊的探索性訊號」，
但不能直接作為「caption 品質已足夠」的文獻門檻。
```

原因有三個：

1. BERTScore 原論文主要證明的是「與 human judgment 的 correlation」，不是提供 universal threshold。
2. 原論文中 image captioning 的 COCO 實驗報的是 Pearson correlation，不是 caption pair 的 raw BERTScore 分數。
3. 原論文唯一明確列出 raw BERTScore 範圍的 Table 11 是 machine translation，不是 image captioning；且 raw score 多落在約 `0.85-0.94`，rescaled score 才落在約 `0.61-0.71`。

因此，如果我們的 report 使用的是 raw `bert-base-multilingual-cased` F1，`0.70-0.75` 反而可能偏低；如果使用的是 rescaled score，則可能接近原論文 MT model 的 rescaled 區間。

2026-07-03 更新：目前 script 會把 BERTScore library output 另行標準化：

```text
normalized = clamp((raw + 1) / 2, 0, 1)
```

所以報告中的主欄位 `bertscore_f1` 是 normalized score；原始值保存在 `raw_bertscore_f1`。

但我們目前 script 仍預設：

```text
rescale_with_baseline = False
```

所以目前分數不是原論文 baseline-rescaled `F̂BERT`。不宜拿 Table 11 的 rescaled `F̂BERT` 直接當門檻。

## 2. 原論文說 BERTScore 在算什麼

原論文 Section 3, PDF p.4 說明：

- BERTScore 使用 contextual embeddings。
- reference token 與 candidate token 之間用 cosine similarity。
- 每個 token 以 greedy matching 找另一句中最相似 token。
- 再計算 precision、recall、F1。

paper 位置：

```text
Section 3 BERTScore, PDF p.4
```

重點公式：

```text
R_BERT = reference token coverage
P_BERT = candidate token support
F_BERT = harmonic mean of P_BERT and R_BERT
```

對本專案的意義：

```text
BERTScore 高，表示 candidate/reference 文字 embedding 相似。
它不代表影像描述醫學正確，也不代表傷口分類正確。
```

## 3. 原論文對 F1 的建議

原論文 Results, PDF p.7 說明：

```text
precision, recall, F1 在不同設定下可能各自表現最好，但 F1 整體可靠，因此作者建議使用 F1。
```

paper 位置：

```text
Section 5 Results, PDF p.7
```

對本專案的意義：

```text
我們用 BERTScore F1 作主數字是合理的。
但 F1 只是 metric choice，不代表 0.70 或 0.75 是通用合格線。
```

## 4. 原論文沒有給「0.75 = good」的通用門檻

原論文的主要實驗設計是：

```text
把 BERTScore 與 human judgment 做 correlation。
```

不是：

```text
規定 BERTScore 達到多少就是 good caption。
```

paper 位置：

```text
Abstract, PDF p.1
Section 4 Experimental Setup, PDF p.5-p.6
Section 5 Results, PDF p.7-p.8
```

原論文 PDF p.1 摘要說，他們用 363 個 machine translation 和 image captioning systems 的輸出做 evaluation，並指出 BERTScore 與 human judgments correlation 更好。

這支持：

```text
BERTScore 適合用於比較系統或排序候選。
```

但不支持：

```text
單一分數 0.70-0.75 可以直接宣稱品質足夠。
```

## 5. 類似任務：Image Captioning 的 COCO 結果

原論文 Section 4, PDF p.6 說明 COCO image captioning 實驗：

- 使用 COCO 2015 Captioning Challenge 的 12 個 submission entries。
- 每個 participating system 會替 COCO validation set 的每張 image 產生一個 caption。
- 每個 image 約有 5 個 human reference captions。
- 用 human judgment 的兩個 system-level metrics：`M1`、`M2`。
- 對 candidate 與多個 references 分別 score，取最高分。

paper 位置：

```text
Section 4 Experimental Setup - Image Captioning, PDF p.6
```

原論文 Table 5, PDF p.8 報的是 Pearson correlation：

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

這裡最重要的解讀：

```text
Table 5 的 0.888、0.917 等是「metric 與 human judgment 的 Pearson correlation」，
不是某張圖片 caption 的 BERTScore。
```

所以不能用 Table 5 說：

```text
我們的 pair-level BERTScore 0.70-0.75 很好或不好。
```

Table 5 能支持的是：

```text
BERTScore 類方法在 image captioning 的 system-level 評估上可與 human judgment 有高 correlation。
```

但它不能提供本專案的 absolute threshold。

### 5.1 COCO image-captioning 計算流程更細節

原論文在 COCO image-captioning 實驗中的資料流可以整理成：

```text
image
-> one candidate caption from a submitted captioning system
-> approximately five human reference captions
-> score candidate against each reference
-> keep the highest BERTScore for that image
-> aggregate to system-level metric score
-> compute Pearson correlation with system-level human M1 / M2
```

也就是說，COCO 實驗不是單一 reference：

```text
BERTScore(candidate, reference_1)
BERTScore(candidate, reference_2)
BERTScore(candidate, reference_3)
BERTScore(candidate, reference_4)
BERTScore(candidate, reference_5)
image_score = max(...)
```

paper 位置：

```text
Section 4 Experimental Setup - Image Captioning, PDF p.6
```

這和本專案目前不同：

| Item | BERTScore paper COCO | Our current project |
|---|---|---|
| Candidate | captioning challenge system output | local Gemma output |
| Reference | 約 5 個 human captions / image | 通常 1 個 SaaS/Gemini baseline JSON |
| Pair scoring | candidate 對每個 reference 算，取最高 | candidate 對指定 reference 算 |
| Human target | system-level M1/M2 | 尚未建立 human acceptance labels |
| Report number | metric-human Pearson correlation | pair/image/category/summary BERTScore |

因此本專案目前的 BERTScore 更接近：

```text
single-reference semantic similarity
```

而不是 COCO 原論文的：

```text
multi-reference captioning system-level correlation
```

這也是為什麼 COCO Table 5 不能直接拿來支持 `0.70-0.75 = good`。

## 6. Raw / Rescaled BERTScore 範圍：Table 11 的限制

原論文 Appendix E, Table 11, PDF p.22 列出公開 MT model 的 BLEU、rescaled BERTScore、raw BERTScore。

paper 位置：

```text
Appendix E, Table 11, PDF p.22
```

Table 11 範例：

| Task | Model | F̂BERT rescaled | FBERT raw |
|---|---|---:|---:|
| WMT14 En-De | ConvS2S | 0.6075 | 0.8488 |
| WMT14 En-De | Transformer-big | 0.6558 | 0.8674 |
| WMT14 En-Fr | ConvS2S | 0.6908 | 0.8841 |
| WMT14 En-Fr | Transformer-big | 0.7061 | 0.8899 |
| IWSLT14 De-En | Transformer-iwslt | 0.6672 | 0.9438 |

這裡可看出：

```text
raw FBERT 常見值約 0.85-0.94。
rescaled F̂BERT 常見值約 0.61-0.71。
```

但 Table 11 是 machine translation，不是 image captioning，也不是 wound captioning。它只能當「raw/rescaled scale 差異」的參考，不能直接當本專案門檻。

## 7. Rescale 對解讀非常重要

原論文 Section 3, Baseline Rescaling, PDF p.5 說明：

- raw BERTScore 理論上是 cosine similarity，範圍可在 `-1` 到 `1`。
- 實際分數常落在較窄範圍。
- 為了 readability，作者用 empirical lower bound 做 baseline rescaling。
- rescaling 目的是讓分數較容易讀，不改變 ranking ability / human correlation。

paper 位置：

```text
Section 3 Baseline Rescaling, PDF p.5
```

對本專案的意義：

```text
同樣是 0.70，raw BERTScore 和 rescaled BERTScore 意義不同。
```

更重要的是：如果我們使用 raw BERTScore，`precision` / `recall` / `F1` 的意思也不是原本 classification 裡的 precision / recall。

它們不是：

```text
precision = TP / (TP + FP)
recall    = TP / (TP + FN)
```

而是：

```text
precision = candidate tokens 對 reference tokens 的平均最大 cosine similarity
recall    = reference tokens 對 candidate tokens 的平均最大 cosine similarity
F1        = 上面兩個 directional similarity 的 harmonic mean
```

paper 位置：

```text
Section 3 BERTScore, PDF p.4
```

也就是說：

```text
raw precision 高
= candidate 產生的 token 大多能在 reference 中找到語意相近的 token support。

raw recall 高
= reference 中的 token 大多能在 candidate 中找到語意相近的 token coverage。

raw F1 高
= support 和 coverage 同時高。
```

但它仍然不是：

```text
70% 正確
70% token overlap
70% wound finding correct
```

所以 `raw BERTScore F1 = 0.70-0.75` 更精確的說法是：

```text
candidate/reference 之間的 contextual token embedding matching 有中等程度相似，
但不能直接轉譯成任務成功率或醫學正確率。
```

如果我們目前 report 是 normalized score：

```text
0.70-0.75 不能拿 Table 11 的 rescaled 0.61-0.71 說「不錯」。
也不能直接當 raw BERTScore 解讀；需回查 raw_bertscore_f1。
```

如果未來開啟：

```text
--rescale-with-baseline
```

則分數尺度會改變，必須重新標註 report。

## 8. 對本專案目前 0.70-0.75 的判斷

目前本專案情境：

```text
candidate: local Gemma wound caption
reference: SaaS/Gemini baseline caption
task: wound image caption / structured visual description
language: English + zh-TW translation comparison
metric model: bert-base-multilingual-cased
rescale: default False
```

若 BERTScore F1 約 `0.70-0.75`，可初步解讀為：

```text
candidate 與 reference 有一定 semantic overlap，但不足以單獨證明 caption 品質足夠。
```

不建議寫：

```text
BERTScore 0.70-0.75 is good.
```

建議寫：

```text
BERTScore 0.70-0.75 is an exploratory semantic-similarity signal.
Because the original BERTScore paper validates correlation rather than absolute thresholds,
we treat this range as requiring per-image evidence review and task-specific calibration.
```

中文報告版：

```text
BERTScore 0.70-0.75 可作為探索性語意相似訊號，但不能直接視為足夠好。
原論文支持 BERTScore 與 human judgment 有較佳 correlation，尤其在 MT / COCO captioning 的 system-level 評估上；
但原論文沒有提供 wound captioning 或 pair-level caption 的 0.70/0.75 合格門檻。
因此本研究需搭配 per-image evidence、visual_summary_only vs full_caption_fields delta、schema validity、VQA accuracy 與人工 review。
```

## 9. 建議本專案如何建立自己的 threshold

下一步不要硬套文獻 threshold，而是建立 project-local calibration。

建議流程：

1. 從現有 run 抽樣，例如每類 2-3 張。
2. 人工標註每個 candidate caption：
   - `good`
   - `acceptable`
   - `bad`
3. 對同一批資料計算：
   - `AC visual_summary_only`
   - `AC full_caption_fields`
   - `BD visual_summary_only`
   - `BD full_caption_fields`
4. 看人工 `acceptable` 的 BERTScore 分布。
5. 再定義本專案自己的 interpretation band。

初始建議 band，可先用於內部討論，不作正式 claim：

| BERTScore F1 | Temporary Interpretation |
|---:|---|
| `< 0.65` | likely weak semantic overlap，需要人工檢查 |
| `0.65-0.75` | moderate overlap，不能自動視為 acceptable |
| `0.75-0.85` | stronger overlap，但仍需看 evidence |
| `> 0.85` | high textual semantic overlap，但仍不代表醫學正確 |

這個 band 是本專案暫定校準規則，不是 BERTScore 原論文門檻。

## 10. 可放進論文 / 簡報的安全說法

建議句：

```text
The original BERTScore paper primarily validates the metric through correlation with human judgments,
not through universal absolute score thresholds. In COCO image captioning, the paper reports
system-level Pearson correlations rather than pair-level BERTScore cutoffs. Therefore, we interpret
our BERTScore values around 0.70-0.75 as exploratory semantic-similarity signals and calibrate them
with per-image evidence, schema validity, VQA-style checks, and human review.
```

中文：

```text
BERTScore 原論文主要證明該 metric 與 human judgment 的相關性，而非提供通用絕對門檻。
在 COCO image captioning 實驗中，原論文報告的是 system-level Pearson correlation，
不是單張 caption pair 的 BERTScore cutoff。因此，本研究將 0.70-0.75 解讀為探索性語意相似訊號，
並搭配逐圖 evidence、schema validity、VQA-style check 與人工 review 進行校準。
```

## 11. Paper Evidence Summary

| Claim | Evidence in Paper |
|---|---|
| BERTScore 是 contextual embedding token similarity，不是醫學正確性指標 | Section 3, PDF p.4 |
| 作者建議用 F1 作為整體穩定的 BERTScore measure | Section 5 Results, PDF p.7 |
| 原論文主要看 human judgment correlation | Abstract PDF p.1; Section 4 PDF p.5-p.6; Section 5 PDF p.7-p.8 |
| COCO image captioning 使用 Pearson correlation，非 absolute BERTScore threshold | Section 4 Image Captioning PDF p.6; Table 5 PDF p.8 |
| COCO Table 5 支持 BERTScore 類方法與 human judgment correlation，但不能支持 0.75 cutoff | Table 5 PDF p.8 |
| raw/rescaled score 尺度不同 | Section 3 Baseline Rescaling PDF p.5 |
| Table 11 raw FBERT 約 0.85-0.94、rescaled F̂BERT 約 0.61-0.71，但任務是 MT | Appendix E Table 11 PDF p.22 |
