"""Compare full-scene Gemma scores with overlapping-window scores on every Censor suite.

Reuse original scores only if the id, text hash and rubric hash match. The model, adapter and
runtime environment must also match the original run. New scores are checkpointed without text.
"""

import argparse
import json
from pathlib import Path
from typing import Any

from evaluation import censor as benchmark
from evaluation import censor_bans, censor_fiction, censor_sft_val
from guardrails.censor import DEFAULT_MODEL, Scores, combine, make_censor, overlapping_windows


def datasets() -> tuple[dict[str, dict[str, str]], dict[str, Any]]:
    samples = benchmark.load_benchmark()
    boundary = [json.loads(line) for line in benchmark.BOUNDARY.read_text().splitlines()]
    groups = censor_bans.cases(json.loads(censor_bans.SUITE.read_text()))
    selected = json.loads(censor_fiction.SELECTION.read_text())
    scenes = {s["scene_id"]: s["text"] for s in map(json.loads, censor_fiction.SCENES.open())}
    refs = [json.loads(line) for line in censor_fiction.REFERENCE.read_text().splitlines()]
    judges = {
        r["id"]: censor_fiction.JudgeVerdict.model_validate(
            {k: r[k] for k in ("children", "teen", "reason")}
        )
        for r in refs
        if r["id"] in selected
    }
    val_texts, val_gold = censor_sft_val.load_val()
    texts = {
        "benchmark": {s.id: s.text for s in samples} | {str(b["id"]): b["text"] for b in boundary},
        "bans": {sid: text for group in groups.values() for sid, text in group},
        "fiction": {sid: scenes[sid] for sid in selected},
        "validation": val_texts,
    }
    return texts, {
        "samples": samples,
        "boundary": boundary,
        "groups": groups,
        "judges": judges,
        "val_gold": val_gold,
    }


def key(sid: str, text: str) -> tuple[str, str]:
    return sid, benchmark._sha(text)


def read_baseline(root: Path, rubric_hash: str) -> dict[tuple[str, str], Scores]:
    paths = [
        root / "censor_eval" / f"{DEFAULT_MODEL}.scores.jsonl",
        root / "censor_fiction" / f"censor.{DEFAULT_MODEL}.jsonl",
    ]
    out: dict[tuple[str, str], Scores] = {}
    for path in paths:
        for row in map(json.loads, path.read_text().splitlines()):
            prompt_hash = row.get("rubric_sha", row.get("prompt_sha"))
            if prompt_hash == rubric_hash:
                out[(row["id"], row["text_sha"])] = Scores.model_validate(
                    row.get("scores", row.get("result"))
                )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--window-chars", type=int, default=2000)
    parser.add_argument("--overlap-chars", type=int, default=400)
    parser.add_argument("--inventory", action="store_true")
    args = parser.parse_args()
    if args.window_chars < 1 or not 0 <= args.overlap_chars < args.window_chars:
        parser.error("invalid window size or overlap")
    texts, context = datasets()
    rubric = (Path("guardrails") / "rubric.md").read_text()
    baseline = read_baseline(args.baseline_root, benchmark._sha(rubric))
    unique = {key(sid, text): text for group in texts.values() for sid, text in group.items()}
    missing = [k for k in unique if k not in baseline]
    long_items = {k: text for k, text in unique.items() if len(text) > args.window_chars}
    print(
        json.dumps(
            {
                "cases": {name: len(group) for name, group in texts.items()},
                "unique": len(unique),
                "baseline_missing": len(missing),
                "windowed_texts": len(long_items),
                "windows": sum(
                    len(overlapping_windows(t, args.window_chars, args.overlap_chars))
                    for t in long_items.values()
                ),
            }
        ),
        flush=True,
    )
    if missing:
        raise RuntimeError("original score cache is incomplete; cannot make a matched comparison")
    if args.inventory:
        return
    from services.mlx_guard import memory_guard
    from writers_room.smoke import watch_free_reserve

    watch_free_reserve(8)
    memory_guard(max_gb=36, min_free_percent=30)
    censor = make_censor(DEFAULT_MODEL)
    args.out.mkdir(parents=True, exist_ok=True)
    cache = args.out / "windows.jsonl"
    done: dict[tuple[str, str], Scores] = {}
    if cache.exists():
        for row in map(json.loads, cache.read_text().splitlines()):
            if (row["window_chars"], row["overlap_chars"], row["rubric_sha"]) == (
                args.window_chars,
                args.overlap_chars,
                benchmark._sha(rubric),
            ):
                done[(row["id"], row["text_sha"])] = Scores.model_validate(row["scores"])
    pred: dict[tuple[str, str], Scores | None] = dict(baseline)
    errors: list[str] = []
    for n, (k, text) in enumerate(long_items.items(), 1):
        if k in done:
            pred[k] = done[k]
            continue
        try:
            parts = [
                censor.classify_text(window)[0]
                for window in overlapping_windows(text, args.window_chars, args.overlap_chars)
            ]
            score = combine(parts)
            pred[k] = score
            row = {
                "id": k[0],
                "text_sha": k[1],
                "window_chars": args.window_chars,
                "overlap_chars": args.overlap_chars,
                "rubric_sha": benchmark._sha(rubric),
                "scores": score.model_dump(),
            }
            with cache.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row) + "\n")
        except Exception as exc:  # noqa: BLE001 - report every failed case
            pred[k] = None
            errors.append(f"{k[0]}: {type(exc).__name__}")
        print(f"windowed {n}/{len(long_items)}; errors={len(errors)}", flush=True)

    def reports(source: dict[tuple[str, str], Scores | None]) -> dict[str, Any]:
        mapped = {
            name: {sid: source.get(key(sid, text)) for sid, text in group.items()}
            for name, group in texts.items()
        }
        fiction, _ = censor_fiction.compare(context["judges"], mapped["fiction"])
        return {
            "benchmark": benchmark.report(
                context["samples"], context["boundary"], mapped["benchmark"]
            ),
            "bans": censor_bans.report(context["groups"], mapped["bans"]),
            "fiction": fiction,
            "validation": censor_sft_val.report(context["val_gold"], mapped["validation"]),
        }

    report = {
        "model": DEFAULT_MODEL,
        "window_chars": args.window_chars,
        "overlap_chars": args.overlap_chars,
        "n_windowed_texts": len(long_items),
        "errors": errors,
        "original": reports(baseline),
        "windowed": reports(pred),
    }
    (args.out / "comparison.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
