# Censor fine-tune: results

October–November 2026. LoRA fine-tunes of local models as the Censor, trained with MLX on the Mac Studio
(`training/censor/train.py`) and scored with `evaluation.censor_check` and `evaluation.censor_sft_val`. Since
2026-10-16 only local models are used, so the goal was a fine-tuned local model that passes every Censor gate.

## Summary

- **Selected: gemma-4-26b-a4b-it (4-bit) + `censor-lora-v3-gemma4_26b_a4b`.** It is the only fine-tune that passes
  every gate, and it passes again with another seed (run B). Training config:
  `training/configs/censor_gemma4_26b_a4b_v3.yaml`. `guardrails/censor.py` still defaults to qwen3.6:27b (Ollama);
  switching the default to this adapter is a separate change.
- **The deciding change was dataset v3.** Every fine-tune on the hand-labelled data alone learned to never set the
  harmful-instruction ban (gemma v1: 4% recall, Qwen3.6 short v1: 0–4%). v3 adds 523 teacher-labelled positives.
- **The recipe is fragile in one way:** gemma runs that trained more at a meaningful learning rate (decay over the
  whole run, a constant LR, or 3 epochs on 4 layers) produce unreadable answers, most of all on long fiction scenes.
- **Both Qwen models** fail the same two gates whatever the recipe: harmful-instruction recall is one sample short
  (89.3%), and they are too lenient on children's fiction (61–69% agreement against 75%).

## Data

- **v1:** 2,466 hand-labelled samples (Gutenberg windows and scenes, an Aegis 2.0 train sample, synthetic
  real-person cases) plus injection-wrapped copies of the positives, split by book or template. Only 5
  harmful-instruction positives.
- **v3 = v1 + teacher labels** (`pipelines/censor_sft/teacher.py`, labels in
  `pipelines/censor_sft/labels/teacher.jsonl`): 2,125 Aegis 2.0 train responses from the categories where real
  instructions occur, labelled without anyone reading them. Qwen3.6-27B base decides `operational_harm`; the best v1
  fine-tune (gemma) gives every other field. 523 positives, matched by the same number of negatives. The teacher set
  the ban on 0.8% of Aegis-"safe" texts (limit 3%) and 31.9% of Aegis-"unsafe" ones. Train 3,846 rows, validation 470
  (`censor_sft_short_v3` for the short prompt: 3,633 / 432).
- Every Censor eval sample is excluded from training and validation by ID, text hash and near-duplicate check.
- Only IDs, text hashes and labels (quotes as character spans) are committed; texts are fetched and assembled at
  build time (`pipelines/censor_sft/build.py`).

## Runs (all on v3 unless noted)

All runs: LoRA rank 8, scale 20, batch 1 with 4-step gradient accumulation, LR 5e-5 with cosine decay to 1e-6 after 30
warm-up updates, max sequence 4,096 (1,024 for the short prompt), prompt masked.

| Run | Model and change | Layers | Epochs | Overall |
|---|---|---|---|---|
| **A** | gemma-4-26b, decay ending early (828 of 961 updates), seed 0 | 16 | 1 | ✅ **all gates** |
| **B** | as A, seed 1 | 16 | 1 | ✅ **all gates** |
| C | gemma-4-26b, decay over the whole run | 16 | 1 | ❌ unreadable answers |
| D | gemma-4-26b, constant LR 1e-6 | 16 | 1 | ❌ ban recall, unreadable answers |
| E | gemma-4-26b, decay over the whole run | 4 | 3 | ❌ ban recall, unreadable answers |
| F | Qwen3-30B-A3B, decay over the whole run | 16 | 1 | ❌ ban recall, children fiction |
| G | Qwen3-30B-A3B, decay ending early (828) | 16 | 1 | ❌ ban recall, children fiction |
| H | Qwen3.6-27B, short prompt with chunking | 16 | 1 | ❌ ban recall, children fiction |
| I | Qwen3.6-27B, short prompt, **v1 data** (no teacher labels) | 16 | 1 | ❌ ban recall, children fiction |

"Decay ending early" is A's schedule: its decay length was sized for an earlier, smaller v3, so the last ~14% of
steps ran at the 1e-6 floor. Qwen3.6-27B fits in memory only with the short prompt.

## Results

**Hard bans** (`evaluation/censor_bans.py`; gates: recall ≥ 90% each, no false flags, no unreadable answers)

| Run | Harmful instructions (n=28) | Real person (n=16) | Injection (n=90) | False flags | Unreadable |
|---|---|---|---|---|---|
| A | 92.9% | 100% | 100% | 0 | 0 |
| B | 92.9% | 100% | 100% | 0 | 0 |
| C | 92.9% | 100% | 98.9% | 0 | **3** |
| D | **64.3%** | 100% | 96.7% | 0 | **1** |
| E | **60.7%** | **81.2%** | 91.1% | 0 | **14** |
| F | **89.3%** | 100% | 96.7% | 0 | 0 |
| G | **89.3%** | 100% | 96.7% | 0 | 0 |
| H | **89.3%** | 100% | 96.7% | 0 | 0 |
| I | **3.6%** | 100% | 94.4% | 0 | 0 |

**Fiction** (`evaluation/censor_fiction.py`, 200 scenes against the frozen claude-opus-5 reference; gates: agreement
children ≥ 75% and teen ≥ 90%, false blocks children ≤ 5% and teen ≤ 2%, unreadable answers ≤ 2%)

| Run | Children agreement | Teen agreement | False blocks (children / teen) | Unreadable |
|---|---|---|---|---|
| A | 77.5% | 97.0% | 0.5% / 0% | 0% |
| B | 80.0% | 96.5% | 1.0% / 0% | 0% |
| C | 80.3% | 95.9% | 2.1% / 0.7% | **26.5%** |
| D | 80.8% | 97.9% | 0% / 0% | **6.0%** |
| E | 80.8% | 96.6% | 1.1% / 0% | **11.5%** |
| F | **68.5%** | 97.5% | 0% / 0% | 0% |
| G | **61.5%** | 97.5% | 0% / 0% | 0% |
| H | **67.3%** | 97.0% | 0.5% / 0% | 0.5% |
| I | **61.0%** | 96.5% | 1.0% / 0% | 0% |

Agreement is measured on readable answers only. The Qwen models' misses are mostly the reference's "warn" scenes rated
"allow" (55–73 scenes, against 24–32 for A and B).

**OpenAI moderation benchmark** (`evaluation/censor.py`, 1,680 samples relabelled for `teen`; gated on sexual recall
≥ 95%, boundary agreement ≥ 80% and unreadable answers ≤ 2%; the rest is for reference)

| Run | Hard-ban recall | Sexual recall | Boundary agreement | Block recall | False blocks | Unreadable |
|---|---|---|---|---|---|---|
| A | 97.8% | 100% | 90.0% | 89.1% | 3.5% | 0% |
| B | 97.2% | 100% | 85.0% | 88.2% | 3.2% | 0% |
| C | 96.9% | 98.3% | 90.0% | 87.2% | 4.0% | 0.8% |
| D | 97.8% | 100% | 82.5% | 84.2% | 3.5% | 0% |
| E | 97.2% | **85.2%** | 87.5% | 84.6% | 8.3% | **8.5%** |
| F | 95.6% | 99.6% | 87.5% | 81.8% | 2.9% | 0% |
| G | 96.0% | 99.6% | 82.5% | 84.0% | 3.4% | 0% |
| H | 95.6% | 100% | 87.5% | 81.0% | 2.2% | 0.1% |
| I | 96.6% | 100% | 90.0% | 85.7% | 3.3% | 0% |

**Validation split** (298 hand-labelled v1 validation texts, not a gate)

| Run | Unreadable | Children agreement | Teen agreement |
|---|---|---|---|
| A | 0% | 93.3% | 94.0% |
| B | 0% | 93.0% | 95.6% |
| C | 2.7% | 91.6% | 92.6% |
| D | 0.7% | 88.6% | 92.6% |
| E | 12.8% | 82.9% | 83.9% |
| F | 0% | 88.6% | 93.6% |
| G | 0% | 87.6% | 91.3% |
| H | 0% | 90.3% | 94.6% |
| I | 0% | 92.3% | 94.3% |

## Lessons

- **Seed the training order.** `train_model` in mlx-lm never seeds NumPy, which it shuffles with, so runs before
  commit `72b1f33` used an unrecorded order and a resumed run drew a new one. D, H and I had been resumed, so they were
  deleted and rerun with the seeded order; the numbers above are from the reruns. None of their conclusions changed.
- **Unreadable answers are failures.** The fiction check used to drop them and measure agreement on the rest, so run C
  passed with 53 of 200 scenes unreadable. It now fails above 2%.
- **Guard evaluation as well as training.** After the Mac froze on 2026-10-30, an evaluation waited three days on a
  stuck MLX call. Evaluation now uses the same memory guard as training and stops a call that runs past 15 minutes.
- **Qwen3.6-27B training is stopped often** by the memory guard (the Mac below 20% free); with the seeded order a
  resume no longer changes the result, only the time taken (about 20 hours per epoch with stops).
