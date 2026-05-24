# 523 Record Summary

Source run: `runs/510_smoke_v1`

This file reorganizes the recorded 510 smoke-test metadata into report-ready tables.

## 1. 欄位說明

| 欄位 | 說明 |
| --- | --- |
| run id | 每次實驗的識別名稱；此報告使用 run directory 名稱。 |
| dataset split | small test / smoke 或 full；目前為 10% smoke test。 |
| image count | 該 provider + prompt type 實際產出的 JSON 檔案數量。 |
| model type | SaaS 或 local。 |
| model | Gemini / Gemma / Qwen 等實際記錄到輸出檔或 experiment record 的模型名稱。 |
| model size / params | 模型參數量；Gemma 使用 Google 文件，Gemini 未公開則不推測。 |
| prompt type | simple / full / chain-style；目前 510 smoke 已記錄 simple 與 full。 |
| platform | server / local / VM；目前以 experiment_record 的 machine 與 OS 表示。 |
| GPU capacity / total VRAM | 硬體規格，不是 runtime usage；pipeline 目前記錄 GPU 型號與總 VRAM。 |
| runtime VRAM usage | 模型實際執行時使用的 VRAM；SaaS 不適用，local 目前未記錄。 |
| runtime GPU usage | 模型實際執行時 GPU utilization；SaaS 不適用，local 目前未記錄，無法事後由輸出檔計算。 |
| input token | API 記錄到的輸入 token；目前只有 Gemini SaaS 有紀錄。 |
| output token | API 記錄到的輸出 token；目前只有 Gemini SaaS 有紀錄。 |
| duration time | 該列輸出檔 metadata 的 total_duration_sec 加總。 |
| prefill time | 若 local runtime 已記錄才填；目前未記錄。 |
| decode time | 若 local runtime 已記錄才填；目前未記錄。 |
| output format quality | JSON schema valid rate 與 required field completion rate。 |
| score 1-4 | 四大文字相似度分數：BLEU-4、ROUGE-L、METEOR-lite、CIDEr-lite。 |
| 備註 | 錯誤、quota、異常狀況、manual review 狀態。 |

## 2. 已記錄實驗資料總表

| run id | dataset split | image count | model type | model | model size / params | prompt type | platform | GPU capacity / total VRAM | runtime VRAM usage | runtime GPU usage | input token | output token | duration time | prefill time | decode time | output format quality | score 1-4 | 備註 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 510_smoke_v1 | small test / smoke 10% (seed 42; 43/431 images) | 43 | local | gemma4:e2b | Gemma 4 E2B effective params | simple | server (baisp-4090, Linux) | NVIDIA GeForce RTX 4090; 24.0 GB | not recorded; cannot be recovered after run | not recorded; cannot be calculated from existing output | not recorded | not recorded | 194.6 sec (3.2 min) | not recorded | not recorded | JSON schema valid 100.0%; field completion 100.0% | BLEU-4 0.046; ROUGE-L 0.297; METEOR-lite 0.380; CIDEr-lite 0.211 | local token / prefill / decode metrics are not recorded in current output files; baseline manual review pending; score status is exploratory |
| 510_smoke_v1 | small test / smoke 10% (seed 42; 43/431 images) | 43 | local | gemma4:e2b | Gemma 4 E2B effective params | full | server (baisp-4090, Linux) | NVIDIA GeForce RTX 4090; 24.0 GB | not recorded; cannot be recovered after run | not recorded; cannot be calculated from existing output | not recorded | not recorded | 606.6 sec (10.1 min) | not recorded | not recorded | JSON schema valid 100.0%; field completion 100.0% | not evaluable; matched image count = 0 | local token / prefill / decode metrics are not recorded in current output files; SaaS full reference missing, so score is not official |
| 510_smoke_v1 | small test / smoke 10% (seed 42; 43/431 images) | 43 | SaaS | gemini-3.1-flash-lite | not published by Google / provider managed | simple | not applicable (SaaS runs on provider infrastructure) | not applicable (SaaS runs on provider infrastructure) | not applicable (SaaS runs on provider infrastructure) | not applicable (SaaS runs on provider infrastructure) | 75,877 | 19,417 | 896.5 sec (14.9 min) | not recorded | not recorded | JSON schema valid 100.0%; field completion 100.0% | reference baseline; no candidate score | request/quota/token tracked: requests 46, success 43, retry 3, rate-limit 0, quota 0; baseline manual review pending; score status is exploratory |
| 510_smoke_v1 | small test / smoke 10% (seed 42; 43/431 images) | 0 | SaaS | gemini-3.1-flash-lite | not published by Google / provider managed | full | not applicable (SaaS runs on provider infrastructure) | not applicable (SaaS runs on provider infrastructure) | not applicable (SaaS runs on provider infrastructure) | not applicable (SaaS runs on provider infrastructure) | 0 | 0 | 0 sec | not recorded | not recorded | no output to validate | not scored; SaaS full reference missing | request/quota/token tracked: requests 46, success 43, retry 3, rate-limit 0, quota 0; not run yet; same-mode full score cannot be calculated |

## 3. Model Size Source Notes

| Model family | Report rule |
| --- | --- |
| Gemma 4 | Use Google Gemma docs: E2B, E4B, 31B, and 26B A4B; A4B means 4B active parameters per token. |
| Gemini | Google Gemini API docs list model families/capabilities but do not publish parameter counts, so this report records provider managed / not published. |

## 4. Dataset Split / Category Count

| Scope | Image Count |
| --- | --- |
| small test / smoke | 43 |
| full / formal | 431 |

| Category | Smoke Count | Full/Formal Count |
| --- | --- | --- |
| abrasions | 9 | 85 |
| bruises | 12 | 122 |
| burns | 6 | 59 |
| cut | 5 | 50 |
| ingrown_nails | 3 | 31 |
| laceration | 6 | 61 |
| stab_wound | 2 | 23 |
