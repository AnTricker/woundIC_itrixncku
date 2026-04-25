# 醫療影像描述開發計畫：VLM 模型微調

## 1. 數據增強 (Data Augmentation)
由於原始資料僅有「Type」標籤，必須先進行「數據提純」以符合 Captioning 需求。

### 1.1 合成標籤 (Synthetic Captioning)
- **作法**：利用 GPT-4o 或 Claude 3.5 對這 100 張圖進行初步醫學打標，生成詳細的 JSON 描述檔。
- **產出**：建立 (Image -> Detailed Text) 的訓練對。

## 2. 模型架構與工具
- **基座模型**：建議選用 HuggingFace 上的 **Moondream2** (1.6B) 或 **LLaVA-v1.6-7B**。
- **訓練工具**：使用 **Unsloth** 或 **LlamaFactory**，這些工具支援 **QLoRA**，能在消費級 GPU 上完成微調。

## 3. 實作流程
1. **轉換格式**：將數據轉為訓練專用的 `jsonl` 格式。
2. **訓練設定**：專注於 Decoder 的微調，讓模型學會醫學語法與描述邏輯。
3. **匯出部署**：訓練完成後，將模型轉換為 GGUF 格式並載入 **Ollama** 執行。

## 4. 關鍵優勢
- 內化領域知識：減少對長 Prompt 的依賴。
- 語氣穩定性：生成的描述會比純 Prompt 方式更具醫療專業感。