# Score Calculation Guide

本文件說明目前 `pipeline.py` 中四個文字相似度分數的實際計算方式：

- `BLEU-4`
- `ROUGE-L`
- `METEOR-lite`
- `CIDEr-lite`

這份說明以目前程式實作為準，不是論文或官方套件的完整版本。程式位置主要在 `pipeline.py`：

- `tokenise`
- `caption_text`
- `bleu4`
- `rouge_l`
- `meteor_lite`
- `cider_lite`
- `paired_scores`

## 1. 分數用途

目前 score 用來比較：

```text
candidate = local model output
reference = SaaS/Gemini baseline output
```

主要比較項目：

```text
local simple vs saas simple
local full   vs saas full
```

目前 simple 的標準 reference 來源是：

```text
runs/saas_simple_baseline
```

也就是說，未來重跑 local simple 時，不需要每次重新呼叫 Gemini simple；score 會拿 local simple 去對這份固定 SaaS simple baseline。

## 2. 先配對圖片

程式先用 JSON 檔名 stem 配對 candidate 與 reference。

範例：

```text
runs/523_smoke_v2/outputs/local/simple/bruises (16).json
runs/saas_simple_baseline/bruises (16).json
```

這兩個檔案的 stem 都是：

```text
bruises (16)
```

所以它們會被視為同一張圖的 candidate/reference pair。

計算方式：

```python
common = sorted(set(candidate_bundles) & set(reference_bundles))
unmatched = len(set(candidate_bundles) ^ set(reference_bundles))
```

意思是：

- `matched_image_count`: candidate 與 reference 都存在的圖片數。
- `unmatched_image_count`: 只存在其中一邊的圖片數。
- 四個 score 只會在 matched images 上計算。
- unmatched images 不會直接拉低 BLEU/ROUGE/METEOR/CIDEr，但會被記錄在 score table 裡。

如果完全沒有 matched image：

```text
BLEU-4 = 0
ROUGE-L = 0
METEOR-lite = 0
CIDEr-lite = 0
```

## 3. 每張圖拿哪些文字來比

程式不是拿整個 JSON 來比，也不是只拿 `visual_summary`。

目前 `caption_text(bundle)` 會抽取以下欄位，接成一大段文字：

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

其中 `observed_supporting_features` 是 list，會用空白接起來。

不參與這四個文字 score 的欄位包括：

- `body_site`
- `wound_presence`
- `measurement_tool`
- `foreign_material_debris`
- `necrosis_eschar`
- `expected_but_not_observed`
- `differential_visual_conflicts`
- `uncertainty`
- `safety_scope`
- `auxiliary_vqa`
- `self_check`
- `metadata`
- `schema_validation`

這代表目前四個 score 主要評估「核心視覺描述與傷口特徵描述」的文字相似度。

## 4. Tokenization 規則

所有分數都先使用同一個 `tokenise(text)`。

目前規則：

```python
re.findall(r"[a-zA-Z0-9_]+", text.lower())
```

意思是：

- 全部轉小寫。
- 只保留英文字母、數字、底線。
- 標點符號會被移除。
- 空白只用來分隔 token。
- 中文字不會被這個 regex 抓到。

重要限制：

```text
目前 score 應該用原始英文 model output 計算，不應該用翻譯後的繁中 output 計算。
```

原因是中文翻譯文字在目前 tokenization 下幾乎不會被正確切成 token，會導致 score 失真。

範例：

```text
"Raw, erythematous surface with serous exudate."
```

會變成：

```text
["raw", "erythematous", "surface", "with", "serous", "exudate"]
```

## 5. BLEU-4 怎麼算

### 5.1 目的

BLEU-4 看 candidate 裡的 n-gram 有多少也出現在 reference 裡。

它偏向回答：

```text
candidate 有沒有使用和 reference 相似的詞組？
```

### 5.2 n-gram

程式會計算 1 到 4 gram：

```text
1-gram: 單字
2-gram: 連續兩個 token
3-gram: 連續三個 token
4-gram: 連續四個 token
```

例如：

```text
tokens = ["raw", "red", "wound", "bed"]
```

則：

```text
1-gram: raw, red, wound, bed
2-gram: raw red, red wound, wound bed
3-gram: raw red wound, red wound bed
4-gram: raw red wound bed
```

### 5.3 每個 n 的 precision

對每個 n：

```text
precision_n = clipped_overlap_count / candidate_ngram_count
```

`clipped_overlap_count` 的意思是：

如果 candidate 某個 n-gram 出現很多次，但 reference 只出現一次，最多只算 reference 的次數。

程式：

```python
overlap = sum(min(count, ref_ngrams[gram]) for gram, count in cand_ngrams.items())
precision = overlap / sum(cand_ngrams.values())
```

如果 candidate 沒有某個 n-gram，程式給一個極小值：

```text
1e-9
```

避免 log 計算直接壞掉。

### 5.4 brevity penalty

BLEU 會懲罰太短的 candidate。

程式：

```python
brevity = 1.0 if len(cand) > len(ref) else exp(1 - len(ref) / len(cand))
```

意思是：

- candidate 比 reference 長：不懲罰，`brevity = 1`
- candidate 比 reference 短或一樣長：給懲罰

### 5.5 最終 BLEU-4

程式使用四個 precision 的幾何平均，再乘上 brevity penalty：

```text
BLEU-4 = brevity * exp((log(p1) + log(p2) + log(p3) + log(p4)) / 4)
```

### 5.6 解讀

高 BLEU-4 表示：

- candidate 與 reference 使用很多相同詞組。
- 特別是 2-gram 到 4-gram 也相似。

低 BLEU-4 可能表示：

- 描述內容不同。
- 用詞不同但意思可能相近。
- candidate 太短。
- 長句順序不同。

注意：

```text
BLEU-4 對同義詞不友善。
```

例如 `redness` 和 `erythema` 語意接近，但 token 不同，BLEU 不會把它們當成相同。

## 6. ROUGE-L 怎麼算

### 6.1 目的

ROUGE-L 使用 LCS，Longest Common Subsequence，最長共同子序列。

它偏向回答：

```text
candidate 和 reference 有沒有保留相似的描述順序？
```

### 6.2 LCS 是什麼

LCS 是兩段 token sequence 中，順序一致但不要求連續的最長共同 token 序列。

範例：

```text
candidate: ["raw", "red", "wound", "surface"]
reference: ["raw", "moist", "red", "surface"]
```

共同子序列可以是：

```text
["raw", "red", "surface"]
```

長度為 3。

### 6.3 程式如何計算 LCS

程式用 dynamic programming 建一個表：

```python
dp = [[0] * (len(ref) + 1) for _ in range(len(cand) + 1)]
```

對每個 candidate token 與 reference token：

```python
if cand_token == ref_token:
    dp[i][j] = dp[i - 1][j - 1] + 1
else:
    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
```

最後：

```python
lcs = dp[-1][-1]
```

### 6.4 precision / recall / F1

程式接著計算：

```text
precision = LCS長度 / candidate token數
recall    = LCS長度 / reference token數
```

最後用 F1：

```text
ROUGE-L = 2 * precision * recall / (precision + recall)
```

如果 precision + recall = 0，則分數為 0。

### 6.5 解讀

高 ROUGE-L 表示：

- candidate 和 reference 有長段相同 token。
- 文字順序也相對接近。

低 ROUGE-L 可能表示：

- 內容差異大。
- 用詞差異大。
- 同樣資訊被用不同順序表達。

ROUGE-L 比 BLEU 更能容忍中間插入一些詞，但仍然不理解同義詞。

## 7. METEOR-lite 怎麼算

### 7.1 目的

這裡的 METEOR-lite 是非常簡化版，只看 token overlap，不做 stemming、synonym、chunk penalty。

它偏向回答：

```text
candidate 和 reference 共享多少關鍵 token？
```

### 7.2 overlap

程式用 Counter 計算 candidate 與 reference 的共同 token 數：

```python
overlap = sum((Counter(cand) & Counter(ref)).values())
```

`Counter(cand) & Counter(ref)` 會取每個 token 的最小出現次數。

範例：

```text
candidate: red red wound
reference: red wound wound
```

共同 token 數：

```text
red: min(2, 1) = 1
wound: min(1, 2) = 1
overlap = 2
```

### 7.3 precision / recall

```text
precision = overlap / candidate token數
recall    = overlap / reference token數
```

### 7.4 最終 METEOR-lite

程式公式：

```text
METEOR-lite = 10 * precision * recall / (recall + 9 * precision)
```

這個公式讓 recall 權重比 precision 更高。

換句話說：

```text
有沒有覆蓋 reference 的內容，比 candidate 是否精簡更重要。
```

如果 precision + recall = 0，分數為 0。

### 7.5 解讀

高 METEOR-lite 表示：

- candidate 覆蓋 reference 中很多 token。
- 特別是 reference 的重要詞有被 candidate 提到。

低 METEOR-lite 可能表示：

- candidate 漏掉很多 reference 詞。
- candidate 用了完全不同的詞。

注意：

```text
目前 METEOR-lite 不處理同義詞。
```

例如：

```text
bruise / contusion
redness / erythema
fluid / exudate
```

如果 token 不同，目前就不算 overlap。

## 8. CIDEr-lite 怎麼算

### 8.1 目的

這裡的 CIDEr-lite 也是簡化版。它不是完整 CIDEr-D，也沒有 TF-IDF corpus weighting。

目前計算方式是：

```text
1 到 4 gram 的 cosine similarity 平均
```

它偏向回答：

```text
candidate 與 reference 的 n-gram 分布是否接近？
```

### 8.2 每個 n 先建立 n-gram count vector

對 n = 1, 2, 3, 4：

```python
cand_ngrams = ngrams(cand, n)
ref_ngrams = ngrams(ref, n)
```

每個 n-gram 的出現次數就是 vector 的值。

範例：

```text
candidate 2-gram counts:
raw red: 1
red wound: 1
wound bed: 1
```

### 8.3 cosine similarity

對每個 n，程式計算 candidate vector 與 reference vector 的 cosine similarity：

```text
cosine = dot(candidate, reference) / (||candidate|| * ||reference||)
```

程式：

```python
dot = sum(cand_ngrams[key] * ref_ngrams[key] for key in keys)
cand_norm = sqrt(sum(value * value for value in cand_ngrams.values()))
ref_norm = sqrt(sum(value * value for value in ref_ngrams.values()))
score_n = dot / (cand_norm * ref_norm)
```

如果 candidate 或 reference 沒有該 n-gram，該 n 的分數為 0。

### 8.4 最終 CIDEr-lite

四個 n 的 cosine similarity 平均：

```text
CIDEr-lite = (score_1 + score_2 + score_3 + score_4) / 4
```

### 8.5 解讀

高 CIDEr-lite 表示：

- candidate 和 reference 的 n-gram 分布相似。
- 不只是有共同單字，連短詞組分布也接近。

低 CIDEr-lite 可能表示：

- 詞彙差異大。
- 詞組順序不同。
- candidate 太短，導致高階 n-gram 很少。

注意：

```text
目前 CIDEr-lite 沒有使用整個資料集的 IDF 權重。
```

所以它不會把罕見、重要的醫學詞自動加權得更高。

## 9. 多張圖片如何彙總

每張 matched image 都會各自算四個分數。

最後取平均：

```python
final_bleu4 = sum(image_bleu4) / matched_image_count
final_rouge_l = sum(image_rouge_l) / matched_image_count
final_meteor_lite = sum(image_meteor_lite) / matched_image_count
final_cider_lite = sum(image_cider_lite) / matched_image_count
```

也就是：

```text
每張圖片權重相同。
```

不是依照文字長度加權，也不是依照類別加權。

## 10. 分數範圍

四個分數理論上主要落在：

```text
0 到 1
```

目前 `scores.md` 顯示時使用三位小數，例如：

```text
0.343
```

不是百分比，雖然欄位格式函式叫 `format_pct`。

可粗略理解：

```text
越接近 1：candidate 越像 reference
越接近 0：candidate 越不像 reference
```

但它們不是「醫學正確率」，也不是人工評分。

## 11. 四個分數的差異總結

| Score | 主要看什麼 | 優點 | 限制 |
|---|---|---|---|
| BLEU-4 | candidate n-gram precision + 短句懲罰 | 看詞組是否像 reference | 對同義詞、改寫很不友善 |
| ROUGE-L | 最長共同子序列 F1 | 看順序接近程度 | 仍然只看 exact token match |
| METEOR-lite | token overlap，recall 權重較高 | 比 BLEU 更重視覆蓋 reference | 沒有 synonym/stemming/chunk penalty |
| CIDEr-lite | 1-4 gram count vector cosine 平均 | 看 n-gram 分布接近程度 | 沒有真正 CIDEr 的 TF-IDF weighting |

## 12. 本專案目前解讀建議

建議不要只看單一 score。

比較合理的解讀方式：

```text
BLEU-4 高：文字和 reference 的片語很像。
ROUGE-L 高：描述順序和 reference 接近。
METEOR-lite 高：reference 的重要 token 覆蓋較多。
CIDEr-lite 高：整體 n-gram 分布較接近。
```

如果四個分數都提升：

```text
可以初步認為 local output 更接近 SaaS reference。
```

如果只有 METEOR-lite 高，但 BLEU-4 低：

```text
可能 candidate 有提到類似關鍵詞，但句型、詞組或順序和 reference 不同。
```

如果 BLEU-4 很低，但人工看起來合理：

```text
可能是同義詞、改寫、描述粒度不同造成，不一定代表醫學描述錯。
```

## 13. 目前重要限制

目前四個 score 都是 lexical score，也就是文字表面相似度。

它們不能直接回答：

```text
這個模型是否醫學上正確？
是否抓到所有關鍵傷口特徵？
是否比 Gemini 更好？
```

它們只能回答：

```text
local output 和 SaaS reference 在目前抽取欄位上的文字相似程度。
```

因此正式報告中應該搭配：

- JSON schema valid rate
- required field completion rate
- matched / unmatched image count
- category-level score
- 人工 review examples
- image + visual summary comparison

## 14. 未來可改進方向

如果要讓 score 更接近醫學描述品質，可以考慮：

- 對中文翻譯另寫中文 tokenizer。
- 加入 synonym mapping，例如 `erythema = redness`。
- 加入醫學關鍵詞 checklist score。
- 對 wound category-specific features 做 field-level scoring。
- 使用真正的 CIDEr / SPICE / BERTScore / embedding similarity。
- 分開計算 `visual_summary`、`wound_features`、`category_specific_check` 的子分數。

目前這版 score 適合當作 smoke test 的快速、自動化文字相似度指標；不適合單獨當作最終模型品質結論。
