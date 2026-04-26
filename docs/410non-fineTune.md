# 醫療影像描述開發計畫：免微調與領域知識注入

## 1. 核心邏輯：RAG-for-Vision (視覺檢索增強)
針對本地模型（如 LLaVA 或 Moondream）看不懂專業術語的問題，我們透過「外部知識掛載」的方式，在推理時為模型提供醫學背景知識。

### 1.1 建立領域知識字典 (Prompt Dictionary)
建立一個 JSON 格式的字典，將傷口標籤對應到標準臨床特徵：
- **實作方式**：當輸入標籤為 `ulcer_grade2` 時，自動抓取對應資訊。
- **資訊內容**：定義、特徵（如真皮層暴露、無腐肉）、建議專家用詞。

## 2. 提示工程優化 (Prompt Engineering)
不再使用單一指令，改為「結構化引導」。

### 優化後的 Prompt 結構
> **[醫療參考資訊]**：{definition}。特徵包含：{features}。
>
> **[任務]**：分析此影像，對應上述資訊，描述傷口的邊緣與組織狀態。
>
> **[分析要求]**：
> 1. 識別病灶位置與邊緣狀況（如：紅腫、浸潤）。
> 2. 判定組織組成（如：肉芽組織百分比）。
> 3. **必須**使用專業術語（如：{expert_terminology}）。
>
> **[限制]**：若特徵不符請說明原因。標註「非醫生診斷」。

## 3. 本地模型部署 (Ollama)
- **硬體**：消費級 GPU 即可負擔。
- **推薦模型**：`moondream` (輕量敏捷) 或 `llava:7b-v1.6`。

===============

```mermaid 
graph TD
    A[VQA Pipeline] --> B[1 --> 15/85 evaluation/train]
    A --> C[2 --> Domain Knowledge]
    A --> D[3 --> Prompt engineering]
    A --> E[4 --> Evaluation]

    %% Domain Knowledge Branch
    C --> C1[generate / collect medical knowledge]
    C1 --> C1a[general healthcare & each wound type]
    C --> C2[by local model / by LLM / by medical textbook/doc.....]

    %% Prompt Engineering Branch
    D --> D1[3-1 classification : simple prompt + refer image]
    D --> D2[3-2 base template + specific detail prompt]