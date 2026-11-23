# Archivist fine-tuning preparation

The Fable v3 default set is split by book: 1,165 train scenes from 20 books; 157 validation
scenes from books 55, 120, 1450; and 214 test scenes from books 36, 113, 215, 421. The test
set is for the final comparison, not for training or configuration selection. The 20 unavailable
scenes in book 1661 are excluded. Fable labels are silver, not human gold.

Build locally from the selected labels and `data/processed/scenes.jsonl`:

```sh
.venv/bin/python -m pipelines.archivist_sft.build
```

The builder checks IDs, counts, duplicate scene texts, schema and evidence quotes, preserves full
scenes and full targets, and records input hashes in `data/interim/archivist_sft/manifest.json`.
It clears book-specific `known_id` targets and supplies an empty known-character list. Consequently
this dataset does not train the model to use nonempty known-character context. Prompts include the
same guidelines and JSON schema as MLX inference with thinking disabled.

Three provisional configurations in `training/configs/archivist_*.yaml` use the installed Qwen3.6
27B, Qwen3 30B-A3B, and Gemma 4 26B-A4B base models. None uses the existing Censor adapter. They
share a single epoch, prompt masking and a 10,240-token limit; the learning settings are starting
points, not validated hyperparameters.

On the Mac, the installed MLX tokenizers counted all 1,536 complete chats without model weights.
Across models the train medians were 7,240–7,330 tokens and the largest chat was 9,865 tokens.
No chat exceeded 10,240. Fitting the length limit does not mean training fits system memory: an
earlier Qwen3.6 Censor run used significant memory at much shorter lengths.

The launcher re-audits complete chat lengths before loading weights and fails if a train or
validation row exceeds the limit. It copies only train and validation chats into the MLX training
directory. It uses the existing 60 GB MLX and 20% system-free-memory guard; this is a periodic
monitor, not a reservation. `--smoke` trains four steps on the longest train example and validates
on the longest validation example; it writes a separate adapter. Do not run the three models
concurrently. For example, after transferring the repository and prepared dataset to the Mac:

```sh
uv run --group mlx python -m training.archivist.train training/configs/archivist_qwen36.yaml --smoke
```

On 2026-11-13 the first three smoke attempts were stopped during model loading at the 8 GiB free-page
reserve, before any training steps. A smoke pass is necessary but cannot
guarantee a full run will fit. The launcher refuses to overwrite an existing adapter and does not
yet resume interrupted runs.

A subsequent Qwen3 30B-A3B smoke run used `--memory-pressure-only` at the user's request. It
completed all four training iterations and saved an adapter. The recorded macOS pressure level
was 65–94% during the run; the 60 GB MLX allocation limit and 20% pressure stop remained active.
This option relaxes the free-page reserve only for smoke tests; it does not establish that full
training or the other two models will fit. The queued benchmark was stopped for this experiment.

Follow-up on 2026-11-14: Qwen3.6 with MLX compilation disabled still hit the 499,000 Metal-resource
limit with 16 LoRA layers. Eight and four LoRA layers, also with compilation disabled, were stopped
by the 60 GB MLX guard (61.3 GB and 63.0 GB observed). Gemma 4 completed the four-iteration
full-scene smoke test. A pass on four iterations is not evidence of full-run stability.

The experimental `pipelines.archivist_sft.windows` builder records evidence and entity-reference
coverage per original scene. At 3,000 characters with 1,500 overlap, it left 3,772/5,521 training
events and 3,511/8,994 relations without a window containing both evidence and all required entity
evidence. The output is marked diagnostic-only and the launcher rejects it for full training.
It must not be treated as a replacement for the full-scene dataset.
The 16-layer Qwen3.6 diagnostic smoke test on the window data (longest training chat: 7,309
tokens) was also stopped by the 60 GB MLX guard at 60.2 GB, with compilation disabled.
The separate 64-step `--smoke-varied` mode selects 32 full training scenes spread across the
observed token-length distribution; it does not train on test scenes.

The requested 8- and 4-layer Qwen3.6 tests use the same diagnostic windows, compilation disabled,
and the 60 GB MLX and 20% macOS memory-pressure guards. The sequential
`training/archivist/window_layer_tests.sh` waits for the Gemma varied-scene test to exit, then
tests both layer counts independently. On the Mac its log is
`/tmp/story_teller_archivist_prep/fix-tests/window-layer-tests.log`. These tests do not change the
diagnostic-only status or establish acceptable silver-label coverage.

At the user's request, the two window smoke tests were repeated with `MAX_MEMORY_GB=70`; they
retain the macOS memory-pressure stop and still run only as diagnostic smoke tests. Their Mac log
is `/tmp/story_teller_archivist_prep/fix-tests/window-layer-tests-70gb.log`. A 16-layer test at
the same cap is queued after them in `fix-tests/window-layer16-70gb.log` under that staging folder.

`evaluation.archivist` runs each base or adapter model over a selected split, records per-scene
inference time, attempt failures, and micro precision, recall and F1 overall and by book. A failed
extraction counts as missing every reference fact. These are agreement metrics against silver, not
human-verified accuracy: the existing scorer matches attribute keys without checking their values
and events by quote overlap, among other approximations. Review disagreements manually.

`training/archivist/queue.sh` waits for a macOS memory-pressure level of at least 60% before
inference or smoke tests, retaining the 60 GB MLX and 20% system pressure stop during each worker.
These workers allow macOS to reclaim inactive memory and do not enforce an 8 GiB free-page floor.
Before full fine-tuning, it still waits for 32 GiB of free pages (8 GiB reserve plus 24 GiB
headroom) and enforces the 8 GiB free-page floor. It runs the three validation baselines first,
retries the longest-example smoke tests, starts full fine-tuning only if all smoke tests pass, and
then evaluates the adapters on validation and both bases and adapters on test.
The queue stops on a worker failure and is not reboot-persistent; completed inference rows can be
resumed, but interrupted training cannot. On the Mac the staging directory is
`/tmp/story_teller_archivist_prep` and progress is in its `queue-reclaim.log`. The three validation
baselines completed; the queue stopped at the Qwen3.6 smoke test. No full fine-tuning has completed.
The provisional configurations remain unvalidated.

On 2026-11-14 the user chose to skip Qwen3.6 and start full-scene Gemma 4 and Qwen3-30B-A3B
fine-tunes, in that order. `training/archivist/two_model_train.sh` runs them sequentially and
independently, then evaluates successful adapters on validation and compares them with their bases
on test. Each training worker retains the 60 GB MLX limit, macOS pressure stop, and 8 GiB free-page
floor; it waits for 32 GiB of free pages before starting. Outputs, checkpoints and `queue.log` are
saved under `/Users/tannguyen/workspace/story_teller_archivist_runs/20261114` on the Mac. The queue
is not reboot-persistent, and a stopped training run cannot yet resume automatically. The two
configs use 1,168 iterations: the 1,165-scene training set plus three repeated scenes complete the
last four-step gradient accumulation.

The separate `archivist_gemma4_l24.yaml` experiment is queued behind the original comparison.
Its guarded queue first runs a longest-scene smoke test and a 64-step varied-scene stability test;
only if both pass will it start full 24-layer training and evaluation. It uses a distinct adapter
and output folder, `/Users/tannguyen/workspace/story_teller_archivist_runs/20261114-gemma24`, on the
Mac. Its process queue is not reboot-persistent.

On 2026-11-15 a separate 20-layer Gemma run was started with the same full-scene data and smoke,
stability, training and evaluation gates. Its Mac output directory is
`/Users/tannguyen/workspace/story_teller_archivist_runs/20261115-gemma20`. As with the 24-layer run,
the smoke tests use memory pressure and the MLX cap; full training additionally enforces the
8 GiB free-page floor, which stopped the previous 24-layer full run.

A separate 8-layer Gemma experiment was started on 2026-11-15, using the same data and gates.
Its Mac output directory is `/Users/tannguyen/workspace/story_teller_archivist_runs/20261115-gemma8`.
It passed both smoke tests, but full training stopped after the latest logged iteration 20 at the
8 GiB free-page floor. On 2026-11-16 the user requested a lower floor; a fresh retry uses 4 GiB,
retains the 60 GB MLX limit and 20% system-pressure stop, and verifies the previous smoke artifacts
match the training config and dataset manifest. Its separate Mac output directory is
`/Users/tannguyen/workspace/story_teller_archivist_runs/20261116-gemma8-reserve4`.

The 20- and 24-layer experiments were queued behind that run on 2026-11-16. Both repeat the
longest-scene and 64-step smoke tests with a 4 GiB free-page floor and retain the 60 GB MLX and
macOS pressure guards. Each full fine-tune runs only if both smoke tests pass. The separate Mac
output directories are `20261116-gemma20-reserve4` and `20261116-gemma24-reserve4` under
`/Users/tannguyen/workspace/story_teller_archivist_runs`. Each job waits for the preceding job,
including its evaluation, and these queues do not survive a reboot.

Qwen3-30B-A3B was queued after the 24-layer Gemma experiment on 2026-11-16. It uses a 4 GiB
free-page floor, a 60 GB MLX cap, fresh full-scene smoke tests, and a new run directory
`/Users/tannguyen/workspace/story_teller_archivist_runs/20261116-qwen3-reserve4` on the Mac. It
starts training from scratch because the older iteration-200 checkpoint has no restorable optimizer
state; that checkpoint is preserved. Validation and held-out test evaluation follow a successful
full fine-tune.
