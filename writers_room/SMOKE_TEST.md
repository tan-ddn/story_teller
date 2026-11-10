# Local writer smoke test

This prepares one chapter using `Qwen3.6-27B-4bit` through MLX and screens
each scene with the project's default fine-tuned Censor. It exercises the existing context builder,
but it is not a complete Showrunner run: Archivist, Scripty and a persistent Story Bible are not
implemented, so the result is an unverified draft. Only planned character attributes are supplied
as facts. The sample plan is intentionally unapproved.

## Before running

- Run on the Mac Studio, where the default MLX Censor model and adapter are installed, with the
  project's Python environment and MLX dependencies available. Set `MLX_MODELS_DIR` and
  `MLX_ADAPTERS_DIR` if their locations differ from the defaults in `services/llm.py`.
- Ensure `Qwen3.6-27B-4bit` is present in `MLX_MODELS_DIR` on the Mac.
- Copy `writers_room/smoke_plan.json`, review or edit it, and set `approved` to `true` in the copy.
  This is the author's approval gate. Select a new or empty output directory.
- The writer and Censor may both occupy memory. Actual throughput and memory use must be measured
  on the Mac; this smoke test records drafting time and tokens, not peak memory.
- Before each model phase the smoke test requires 48 GiB of free macOS pages: a 24 GiB reserve
  for other users plus 24 GiB of estimated model headroom. Adjust with `--reserve-gib` and
  `--model-headroom-gib`. These are snapshots and estimates, not a hard memory limit or a monitor;
  other workloads or model allocations can consume memory after a check. No other user's models
  or processes are stopped. A failed second check discards the draft before Censor screening.
- The process also runs the existing MLX monitor with a 36 GB allocation ceiling and a minimum
  macOS memory-pressure availability level of 30%. It checks periodically, so short spikes can
  exceed a threshold. The macOS availability level and free pages are different measurements.
- The writer runs in a separate process and transfers unscreened prose through a pipe. Its process
  exits before the second memory check and Censor load, releasing its model memory without writing
  an unscreened draft to disk. Both processes have the same MLX monitor settings.

When ready, from the repository root on the Mac:

```bash
MLX_MAX_TOKENS=4096 uv run --group mlx python -m writers_room.smoke /path/to/approved-plan.json /path/to/new-output-directory
```

An allowed chapter is saved as `ch01.md`; `result.json` records the status, model, word and token
counts, drafting time, and Censor warnings. A blocked or unavailable Censor produces no chapter
file and exits with a nonzero status. It does not attempt a tone-down revision. Output is marked
`continuity_checked: false`; inspect the text manually. Never reuse an output directory containing
an earlier draft.

The isolated unit checks can be run separately when execution is authorized:

```bash
uv run pytest tests/unit/test_writers_room_smoke.py
```
