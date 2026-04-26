# 傷口影像描述 VLM 系統：技術開發文件
**Project: Clinical Wound Image-Captioning for Second Opinion (v2.0)**

## 1. 系統概述 (System Overview)
本系統旨在利用現成的（Off-the-shelf）大型視覺語言模型（VLM）於局部伺服器環境下，針對特定臨床傷口影像生成高質量的「第二意見（Second Opinion）」描述數據。系統採單一模型直接推論架構，透過高度解耦的提示詞工程（Prompt Engineering）確保生成內容的臨床參考價值。

## 2. 技術棧與環境配置 (Technology Stack)
* **運算硬體**：RTX 4090 24GB VRAM (Server)。
* **推論引擎**：Ollama API 整合。
* **核心模型**：
    * **Local Models**：LLaVA-v1.6-34B、Qwen 3.5、Gemma 4 (用於效能對比)。
    * **Baseline (SaaS)**：Gemini、GPT+ (作為標竿評分基準)。
* **開發語言**：Python 3.10+。

## 3. 數據架構與檔案管理 (Data Architecture)
系統採完全解耦設計，避免硬編碼提示詞，便於快速疊代與評估。

### 3.1 目錄結構
```text
/itriVQA_An/
├── images/            # 原始傷口影像 (格式: {category} (n).jpg)
├── prompts/           # 提示詞組件
│   ├── base_template.md   # 通用描述框架與約束
│   ├── {category}.md      # 針對 Abrasions, Burns 等之專屬指令
├── output/            # 各模型生成的 JSON 影像描述
└── main.py            # 自動化批次處理腳本
```

### 3.2 支援傷口類型 (Target Categories)
目前系統支援以下 7 種原文定義之分類：
* `Abrasions`, `Bruises`, `Burns`, `Cut`, `Ingrown_nails`, `Laceration`, `Stab_wound`

## 4. 核心功能實作 (Core Implementation)

### 4.1 影像描述生成邏輯 (Captioning Logic)
系統捨棄傳統兩階段 VLM-RAG 流程，改採單一模型端到端（End-to-End）生成。
1.  **檔案映射**：程式讀取影像檔名之類別標籤。
2.  **動態拼接**：將 `base_template.md` 與對應之 `{category}.md` 拼接成完整的 System Prompt。
3.  **單一推論**：VLM 讀取影像並依據 Prompt 直接產出結構化 JSON。

### 4.2 絕對數據校準約束 (Absolute Data Constraints)
為解決模型對傷口尺寸的「通靈」問題，實施嚴格約束：
* **指令**：除非視覺上識別出物理校準工具（如尺標、校準卡），否則 `area_estimate` 欄位必須回傳 `unknown`。
* **目的**：確保數據作為 Baseline 的科學客觀性。

## 5. 評估機制與效能對比 (Evaluation & Benchmarking)

### 5.1 評分基準 (Baseline Construction)
系統以 SaaS LLM (Gemini/GPT+) 的輸出作為最高品質基準線。評分維度包括：
* **描述精準度**：臨床術語使用的正確性。
* **幻覺率**：是否在乾淨傷口中誤報碎屑（Debris）或焦痂（Eschar）。
* **格式遵循度**：JSON 結構的穩定性。

### 5.2 局部模型效能分析 (Local Model Scores)


| 模型 | 強項 | 弱項 |
| :--- | :--- | :--- |
| **Qwen 3.5** | JSON 格式極其穩定，適合批次產出。 | 深度判定（Depth Category）偏保守。 |
| **Gemma 4** | 顏色演進與邊緣分析細膩。 | 易受指令引導（Instruction Bias）產生虛假特徵。 |

## 6. 已知挑戰與技術債 (Technical Debt)
* **誘導式幻覺**：特定類別的 Prompt 引導太強時，模型會傾向於無中生有。
* **2D 深度判定**：模型在判斷 `depth_category` 時，對於全皮層損害與表淺損害的視覺界限模糊。
* **解決方案**：計畫引入「否定性回報（Negative Reporting）」指令，要求模型明確標註「未見特徵」。
