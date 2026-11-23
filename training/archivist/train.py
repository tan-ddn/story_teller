"""Guarded MLX Archivist LoRA launcher. Run only after a successful memory smoke test."""

import argparse
import json
import math
import os
import types
from pathlib import Path
from typing import Any

import yaml

from services.mlx_guard import memory_guard
from training.archivist.token_audit import audit


def varied_scene_ids(entries: list[dict[str, Any]], count: int = 32) -> set[str]:
    """Select examples evenly across the token-length distribution, including both extremes."""
    ordered = sorted(entries, key=lambda row: (row["tokens"], row["scene_id"]))
    count = min(count, len(ordered))
    if count < 2:
        return {str(row["scene_id"]) for row in ordered}
    return {str(ordered[i * (len(ordered) - 1) // (count - 1)]["scene_id"]) for i in range(count)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--smoke-varied",
        action="store_true",
        help="run 64 smoke steps across 32 full scenes of varied lengths",
    )
    parser.add_argument("--max-memory-gb", type=float, default=60.0)
    parser.add_argument(
        "--reserve-gib", type=float, default=8.0, help="minimum macOS free pages; default 8 GiB"
    )
    parser.add_argument(
        "--memory-pressure-only",
        action="store_true",
        help="smoke only: let macOS reclaim inactive pages; retain MLX and pressure guards",
    )
    args = parser.parse_args()
    if not math.isfinite(args.reserve_gib) or args.reserve_gib < 0:
        parser.error("--reserve-gib must be a finite non-negative number")
    if args.memory_pressure_only and not args.smoke:
        parser.error("--memory-pressure-only is permitted only with --smoke")
    if args.smoke_varied and not args.smoke:
        parser.error("--smoke-varied requires --smoke")
    config: dict[str, Any] = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    model_name = config.pop("model")
    adapter_name = config.pop("adapter_name")
    data = Path(config.pop("dataset"))
    manifest = json.loads((data / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("diagnostic_only") and not args.smoke:
        raise ValueError("diagnostic-only data must not be used for full training")
    models = Path(os.environ.get("MLX_MODELS_DIR", "/Volumes/Untitled/Shared/models"))
    adapters = Path(os.environ.get("MLX_ADAPTERS_DIR", "adapters"))
    if args.smoke:
        adapter_name += "-smoke"
        config.update(iters=4, steps_per_report=1, steps_per_eval=4, val_batches=1, save_every=4)
        if args.smoke_varied:
            adapter_name += "-varied"
            config.update(iters=64, steps_per_report=8, steps_per_eval=64, save_every=64)
    output = adapters / adapter_name
    if output.exists():
        raise FileExistsError(f"adapter already exists: {output}; choose a new name")

    from writers_room.smoke import watch_free_reserve

    memory_guard(args.max_memory_gb)
    if not args.memory_pressure_only:
        watch_free_reserve(args.reserve_gib)

    # This compares complete chats using the installed tokenizer; no row is dropped or clipped.
    report = audit(models / model_name, data, config["max_seq_length"])
    for split in ("train", "valid"):
        if report["splits"][split]["over_limit"]:
            raise ValueError(f"{split} examples exceed max_seq_length; inspect token audit")

    # Keep the test set out of the directory passed to the trainer.
    mlx_data = Path(
        "data/interim/archivist_sft_mlx_smoke" if args.smoke else "data/interim/archivist_sft_mlx"
    )
    mlx_data.mkdir(parents=True, exist_ok=True)
    if (mlx_data / "test.jsonl").exists():
        raise ValueError(f"remove test data from training directory: {mlx_data}")
    for split in ("train", "valid"):
        source = data / f"{split}.jsonl"
        with (
            source.open(encoding="utf-8") as src,
            (mlx_data / source.name).open("w", encoding="utf-8") as dst,
        ):
            selected = (
                varied_scene_ids(report["splits"][split]["examples"])
                if args.smoke_varied and split == "train"
                else {report["splits"][split]["longest_scene_id"]}
            )
            for line in src:
                row = json.loads(line)
                if args.smoke and row["scene_id"] not in selected:
                    continue
                dst.write(json.dumps({"messages": row["messages"]}) + "\n")

    import mlx.core as mx
    import numpy as np
    from mlx_lm.lora import CONFIG_DEFAULTS, load, load_dataset, train_model
    from mlx_lm.tuner import trainer
    from mlx_lm.tuner.callbacks import TrainingCallback

    from training.censor.train import _reproducible_order, _thinking_off

    settings = (
        CONFIG_DEFAULTS
        | config
        | {
            "model": str(models / model_name),
            "adapter_path": str(output),
            "data": str(mlx_data),
            "train": True,
            "test": False,
            "report_to": None,
        }
    )
    trainer.iterate_batches = _reproducible_order(
        trainer.iterate_batches, np.random.seed, settings["seed"], 0
    )
    mx.set_cache_limit(4_000_000_000)
    output.mkdir(parents=True)
    (output / "data_manifest.json").write_text(
        (data / "manifest.json").read_text(encoding="utf-8"), encoding="utf-8"
    )
    (output / "token_audit.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (output / "training_config.yaml").write_text(args.config.read_text(), encoding="utf-8")
    (output / "memory_guard.json").write_text(
        json.dumps(
            {
                "mlx_limit_gb": args.max_memory_gb,
                "free_page_reserve_gib": None if args.memory_pressure_only else args.reserve_gib,
                "pressure_stop_percent": 20,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    model, tokenizer = load(settings["model"])
    _thinking_off(tokenizer)
    if "deepseek-r1-distill" in model_name.lower():
        from training.archivist.deepseek_chat import align_deepseek_template

        align_deepseek_template(tokenizer)
    margs = types.SimpleNamespace(**settings)
    train, valid, _ = load_dataset(margs, tokenizer)
    train_model(margs, model, train, valid, TrainingCallback())


if __name__ == "__main__":
    main()
