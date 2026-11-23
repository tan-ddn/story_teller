# Archivist benchmark results, 21 November 2026

The Fable v3 silver set contains 1,536 scenes from 27 books. Training uses 1,165 scenes from 20 books; validation uses 157 scenes from three held-out books; test uses 214 scenes from four different books. Scores measure agreement with silver labels, not independently verified accuracy. Failed schema or verbatim-evidence checks count as empty extractions. Attribute matching does not verify values, and event matching uses quote overlap; interpret results accordingly.

## Fine-tuning setup

The principal runs use rank-8 LoRA on the final 16 language-model layers, full scenes, one training epoch, microbatch size one and gradient accumulation of four. The CUDA runs use 4-bit QLoRA; the Mac runs use MLX 4-bit models. The training and validation books are disjoint. Model-specific token audits check full chat lengths rather than silently truncating scenes. Different tokenizers and LoRA implementations mean that matching the layer count does not make the trained parameters or quantization identical.

Validation uses schema and verbatim-evidence checks, at most two attempts, and category-level micro precision, recall and F1. Failed extractions are scored as empty; the resulting F1 mixes extraction quality with output-format and evidence reliability. Scores for entity, attribute, event, relation and time cues are distinct; no overall F1 is inferred by averaging them. The test set remains separate from validation.

## Validation: completed runs

All rows below use the same 157 validation scenes. F1 values are percentages; median time is seconds per scene including retries. Hardware, quantization and inference implementations differ, so timings are not an apples-to-apples model-speed comparison.

| Model | Configuration | Accepted | Entity F1 | Attribute F1 | Event F1 | Relation F1 | Time-cue F1 | Median s |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Qwen3.6 27B | 16-layer adapter, CUDA | 134/157 | 77.8 | 43.7 | 34.1 | 29.9 | 56.7 | 320.1 |
| Qwen3.8 27B | 16-layer adapter, CUDA | 131/157 | 76.8 | 44.8 | 33.7 | 32.9 | 59.6 | 293.0 |
| Qwen3.8 27B | base, MLX 4-bit | 117/157 | 64.7 | 26.3 | 20.9 | 20.7 | 44.0 | 79.7 |
| Gemma 4 31B | 16-layer adapter, CUDA | 82/157 | 56.4 | 28.2 | 20.8 | 22.9 | 41.4 | 450.9 |
| Gemma 4 26B-A4B | base, MLX 4-bit | 7/157 | 3.2 | 1.8 | 1.4 | 0.3 | 1.6 | 30.8 |
| Gemma 4 26B-A4B | 16-layer adapter, MLX 4-bit | 57/157 | 41.7 | 22.5 | 20.4 | 13.9 | 32.7 | 81.6 |
| Gemma 4 26B-A4B | 8-layer adapter, MLX 4-bit | 10/157 | 5.9 | 2.8 | 3.2 | 2.0 | 5.1 | 153.9 |
| DeepSeek R1 Distill Qwen 14B | base, MLX 4-bit | 16/157 | 6.3 | 2.0 | 1.6 | 0.5 | 1.6 | 71.9 |
| DeepSeek R1 Distill Qwen 14B | converted 16-layer CUDA adapter, MLX 4-bit | 14/157 | 9.7 | 4.8 | 4.5 | 1.5 | 6.9 | 441.4 |
| GLM 4 9B Chat 1M | base, MLX 4-bit | 0/157 | 0 | 0 | 0 | 0 | 0 | 24.7 |
| GLM 4 9B Chat 1M | 16-layer adapter, MLX 4-bit | 0/157 | 0 | 0 | 0 | 0 | 0 | 384.8 |

Qwen3.6 and Qwen3.8 adapters lead the completed runs on acceptance and most extraction categories. Qwen3.8's base and adapted results use different inference stacks and quantization; this limits causal attribution. The original Mac baseline summaries for Qwen3.6 and Qwen3-30B were not recovered. Earlier session notes recorded entity F1 of 20.6% and 6.0% respectively, but not their complete breakdowns. Gemma 26B's baseline was rerun and preserved on 23 November.

The Gemma 26B matched MLX validation pair gained 38.5 percentage points in entity F1 and 50 accepted scenes. Its test pair gained 35.9 percentage points in entity F1 and 67 accepted scenes. Qwen3.6's recorded baseline-to-adapter entity gain is 57.2 points, but its missing full baseline record and changed backend make that comparison less controlled. GLM's 0/157 accepted extractions in both runs indicate a schema/output-path failure, not evidence that its semantic ability is zero.

## Test: Gemma 4 26B-A4B

These 214 scenes are a different split and should not be compared directly to validation rows.

| Configuration | Accepted | Entity F1 | Attribute F1 | Event F1 | Relation F1 | Time-cue F1 | Median s |
|---|---:|---:|---:|---:|---:|---:|---:|
| Base | 15/214 | 3.8 | 1.9 | 3.2 | 1.0 | 5.7 | 29.3 |
| 16-layer adapter | 82/214 | 39.7 | 26.0 | 14.3 | 14.4 | 35.2 | 81.1 |
| 8-layer adapter | 23/214 | 8.5 | 4.5 | 4.0 | 3.3 | 9.4 | 152.5 |

## Incomplete and missing results

- The DeepSeek 14B CUDA extraction benchmark was stopped at 11/157 scenes; the completed DeepSeek comparison above was subsequently run on MLX with a converted adapter. Direct numerical equivalence of the converted and original adapter was not established.
- Qwen3-30B did not complete full fine-tuning. The GLM 4.7 Flash smoke test ran out of GPU memory before training. Both GLM 9B runs failed schema validation on every attempt, so the zero scores warrant output-path investigation before interpreting model quality.
- The Gemma 31B base-model validation benchmark is missing: its Mac retries hit the 8 GiB free-page guard. Its 16-layer run initially ran out of CUDA memory, then resumed from checkpoint 50 using expandable allocator segments and finished.
- The final Gemma 31B adapter was copied locally and verified against the Vast weight checksum. The DeepSeek 14B and GLM 9B model and adapter artifacts were removed at the user's request; their records were retained.

## Data and provenance

The copied [Vast and Mac metrics](../../data/interim/archivist_report_20261121/) include per-scene results, summaries, metadata, training and smoke metrics, and relevant logs. Previously recovered fine-tuned [Qwen3.6 metrics](../../data/interim/archivist_vast_qwen36/) and [Qwen3.8 metrics](../../data/interim/archivist_vast_qwen38_metrics/) remain in their original directories. These interim data directories are Git-ignored and are available only where separately retained. The completed validation runs have the same dataset SHA-256 (`ed0d7ede54f99b8628f7f6ec9b856ec4b6adf632a63cbc7a5239c69b0d552245`).
