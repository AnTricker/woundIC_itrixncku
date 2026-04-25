import ollama
import json
import os
import glob
from datetime import datetime

# 路徑定義
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROMPT_DIR = os.path.join(BASE_DIR, "prompts")
IMAGE_DIR = os.path.join(BASE_DIR, "images")
OUTPUT_DIR = os.path.join(BASE_DIR, "output/qwen35")
MODEL_NAME = "qwen3.5" # 善用 4090 的算力

def load_text(filename):
    path = os.path.join(PROMPT_DIR, filename)
    if not os.path.exists(path): return None
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()

def run_captioning():
    base_template = load_text("base_template.md")
    if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)

    # 遍歷所有影像
    for img_path in glob.glob(os.path.join(IMAGE_DIR, "*.jpg")):
        img_filename = os.path.basename(img_path)
        
        # 從檔名提取類別 (例如 "burns (2).jpg" -> "burns")
        category = img_filename.split(' ')[0].lower()
        
        print(f"🚀 Processing {img_filename} as {category}...")

        # 載入該類別專屬指令
        specific_instructions = load_text(f"{category}.md")
        if not specific_instructions:
            print(f"⚠️ Skip: Missing {category}.md")
            continue

        # 組合最終 Prompt
        final_prompt = base_template.format(
            category=category,
            specific_instructions=specific_instructions
        )

        try:
            # 執行 VQA
            response = ollama.chat(
                model=MODEL_NAME,
                format='json',
                messages=[{
                    'role': 'user',
                    'content': final_prompt,
                    'images': [img_path]
                }]
            )

            # 存檔
            save_name = img_filename.replace(".jpg", ".json")
            with open(os.path.join(OUTPUT_DIR, save_name), 'w', encoding='utf-8') as f:
                f.write(response['message']['content'])
            
            print(f"✅ Baseline saved: {save_name}")

        except Exception as e:
            print(f"❌ Error: {str(e)}")

if __name__ == "__main__":
    run_captioning()