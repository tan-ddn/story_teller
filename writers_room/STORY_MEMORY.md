# Local multi-chapter writing

Run on the Mac Studio with the project MLX environment, an author-approved `StoryPlan` JSON,
`Qwen3.6-27B-4bit` in `MLX_MODELS_DIR`, and the default Censor base model and adapter installed:

```bash
MLX_MAX_TOKENS=4096 uv run --group mlx python -m writers_room.local plan.json output-directory --chapter-gate
```

Each model call runs in its own guarded process. A preflight requires 32 GiB free pages, including
a nominal 24 GiB reserve; every model worker also monitors free pages and stops if they fall
below 24 GiB. The worker has the existing 36 GB MLX and 30% macOS memory-availability monitors.
These are periodic checks, not a hard system-wide reservation, and allocations can spike between
checks or compete with other workloads.
The local Censor serializes checks so scenes cannot load several models concurrently. The planned
Qwen Archivist is an unevaluated baseline; schema or evidence failures stop the run.

For experiments, `--censor-window-chars 2000` screens overlapping character windows instead of
whole scenes, with 20% overlap and a block if any window blocks or fails. The default remains whole
scenes: windowing can change context and has not been validated against the full Censor suite.
It does not window Archivist extraction or reduce the model weights' memory footprint.

The SQLite Bible stores full extractions and accepted text together in one transaction. At the
next run, Showrunner restores a missing chapter file from the database and refuses a conflicting
file. The output directory is bound to the exact approved plan. `ledger.jsonl` remains a separate
write and may omit an accepted attempt after a crash. Chapter-gate interruptions use an in-memory
checkpointer: after a process restart the interrupted draft is regenerated, not recovered.

`facts_about` exposes unhedged main-story claims and supports unambiguous aliases found alongside
an entity. This is not general entity resolution; identical names may refer to different entities.
Facts are ordered by narrative position rather than a fully resolved story timeline. Scripty
reports selected conflicting attributes as low-severity questions for human review, including
flashbacks with uncertain time; it excludes dreams, prophecies, reported stories and hedged
claims. For death-related event flags only, unambiguous ISO scene dates can establish that an
event predates the death; other date formats and unclear chronology remain unproven. It does not
prove contradictions, infer elapsed time, perform plan-drift or factuality checks, or resolve
legitimate changes. Inspect flags in the ledger and review every chapter. Evidence matching checks
that quotations occur in the text; it does not prove extracted claims are supported by them.

Use one process per project directory. Do not reuse a directory for a different plan. Writer,
Censor and Archivist calls require local Apple Silicon models and have not yet been evaluated as
a combined system; unit tests use scripted model responses.
