# 510 Progress Summary

Generated on 2026-05-23 from `runs/510_smoke_v1`.

## Current Status

The 10% smoke-test pipeline has been completed for the `simple` prompt path.

Completed:

- Manifest prepared for `images/`.
- `MM-SkinQA` excluded.
- Smoke scope created with 43 images from 431 supported formal images.
- Local `simple` outputs generated for all 43 smoke images.
- Gemini `simple` SaaS outputs generated for all 43 smoke images using `gemini-3.1-flash-lite`.
- Local `simple` vs SaaS `simple` scoring completed.
- Charts generated under `runs/510_smoke_v1/charts/`.
- `experiment_record.json`, `scores.json`, `scores.csv`, and `scores.md` exist.

Partially completed:

- Local `full` outputs exist for all 43 smoke images.
- SaaS `full` outputs are not generated yet.
- Therefore `local_full_vs_saas_full` cannot be evaluated yet.

## Dataset Split

Total supported formal images: `431`

Smoke images: `43`

Smoke category distribution:

| Category | Smoke Count |
|---|---:|
| abrasions | 9 |
| bruises | 12 |
| burns | 6 |
| cut | 5 |
| ingrown_nails | 3 |
| laceration | 6 |
| stab_wound | 2 |

## Generated Outputs

Current output counts:

| Provider | Prompt Mode | Count |
|---|---:|---:|
| local | simple | 43 |
| local | full | 43 |
| saas | simple | 43 |
| saas | full | 0 |

The completed official score comparison is:

```text
local_simple_vs_saas_simple
```

The missing official score comparison is:

```text
local_full_vs_saas_full
```

because there are no SaaS `full` smoke outputs yet.

## Simple Score Result

From `runs/510_smoke_v1/scores.md`:

| Comparison | Matched | Unmatched | BLEU-4 | ROUGE-L | METEOR-lite | CIDEr-lite |
|---|---:|---:|---:|---:|---:|---:|
| local_simple_vs_saas_simple | 43 | 0 | 0.046 | 0.297 | 0.380 | 0.211 |

All 43 local simple outputs matched a SaaS simple reference.

Quality-table metrics are all currently `1.000` for:

- schema valid rate
- field completion
- not-observed usage
- no-ruler compliance
- safety scope compliance

Important note: the baseline quality gate is still `pending` because manual review has not been marked pass.

## Gemini Probe Findings

Gemini probing showed two separate risks:

1. Image/content blocking.
2. Request-rate limiting.

Probe summary:

| Probe | Seed | Category | Image | Simple | Full |
|---|---:|---|---|---|---|
| `090247` | 42 | laceration | `laseration (57).jpg` | error | error |
| `094257` | 42 | laceration | `laseration (57).jpg` | blocked | blocked |
| `095042` | 43 | abrasions | `abrasions (45).jpg` | ok | ok |
| `095702` | 44 | burns | `burns (56).jpg` | ok | ok |
| `095805` | 45 | bruises | `bruises (44).jpg` | ok | ok |
| `095852` | 46 | abrasions | `abrasions (62).jpg` | ok | ok |
| `095953` | 47 | burns | `burns (20).jpg` | ok | 429 |
| `100206` | 48 | laceration | `laseration (12).jpg` | blocked | 429 |
| `101455` | 42 | laceration | `laseration (57).jpg` | ok | ok |

Interpretation:

- Earlier `gemini-2.5-flash` probes hit `blockReason: OTHER` on laceration images.
- Later `gemini-3.1-flash-lite` successfully processed the same seed-42 laceration image.
- Back-to-back full probes triggered `429 Too Many Requests`.
- Gemini blocking is image/model dependent, not only category dependent.
- Gemini `full` mode is much more likely to hit rate limits because it uses three API calls per image.

## Gemini Simple Run Usage

From `experiment_record.json`, SaaS simple generation used:

| Metric | Value |
|---|---:|
| request_count | 46 |
| api_success_count | 43 |
| retry_count | 3 |
| success_count | 43 |
| failure_count | 0 |
| input_token_count | 75,877 |
| output_token_count | 19,417 |
| total_token_count | 95,294 |
| total_duration_sec | 896.5 |

Meaning:

- `request_count` is API call attempts and quota pressure.
- `success_count` is completed output files.
- `request_count > success_count` because retry attempts also count as requests.
- The simple run completed without final failures.

## Hardware / Runtime Context

Platform recorded:

| Field | Value |
|---|---|
| machine | `baisp-4090` |
| OS | Linux |
| GPU | NVIDIA GeForce RTX 4090 |
| VRAM | 24 GB |
| RAM | 63 GB |

Runtime summary:

- Local generation total recorded duration: about `997.2 sec`.
- SaaS simple generation total recorded duration: about `896.5 sec`.

Note: local generation stats currently include accumulated rerun counts in the record, so output-folder counts are more reliable for final artifact status.

## Current Limitations

The current result is useful but not final.

Limitations:

- SaaS `full` mode has not been run for the full smoke set.
- `local_full_vs_saas_full` score is unavailable.
- Baseline quality gate is still manual-review pending.
- Scores are text-similarity metrics against a SaaS reference, not medical ground truth.
- Gemini may still block some graphic wound images depending on model and image content.
- Repeated Gemini full calls can hit RPM limits unless pacing is conservative.

## Recommended Next Steps

1. Manually review a sample of SaaS simple outputs.
2. If simple baseline quality is acceptable, mark the baseline review result in the experiment notes.
3. Run SaaS full smoke only after planning rate limits carefully.
4. For `gemini-3.1-flash-lite`, use at least:

```bash
--saas-request-delay-sec 4
```

5. Consider running SaaS full by category or in smaller batches to avoid RPM pressure.
6. After SaaS full exists, rerun:

```bash
python -m pipeline score \
  --run-dir runs/510_smoke_v1 \
  --reference-mode full \
  --score-scopes smoke
```

7. Then regenerate charts:

```bash
python -m pipeline charts \
  --run-dir runs/510_smoke_v1
```

## Bottom Line

The simple smoke pipeline is now complete end-to-end:

```text
local simple -> SaaS simple -> same-mode score -> charts/records
```

The main remaining work is to finish the SaaS `full` smoke path and then compute:

```text
local_full_vs_saas_full
```

Until the baseline manual review is passed, the current score should be treated as exploratory rather than final.
