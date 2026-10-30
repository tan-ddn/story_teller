"""LoRA fine-tune of the Censor model with MLX, logged to MLflow ("censor-train").

Reads a YAML config (training/configs/), writes the chat data in the layout mlx-lm expects
(train.jsonl / valid.jsonl), trains with mlx-lm's own trainer and reports losses, learning rate,
throughput and peak memory to MLflow. The finished adapter (small: LoRA weights and its config) is
logged as an artifact. --smoke trains a few steps into "censor-train-smoke" to check memory and
speed first.

Runs on the Mac (mlx). Models come from MLX_MODELS_DIR, adapters go to MLX_ADAPTERS_DIR, the
same folders services.llm uses, so "mlx:<model>+<adapter_name>" evaluates the result.

uv run python -m training.censor.train training/configs/censor_qwen3_30b_a3b_lora.yaml --smoke
uv run python -m training.censor.train training/configs/censor_qwen3_30b_a3b_lora.yaml
"""

import argparse
import itertools
import json
import math
import os
import re
import subprocess  # nosec B404
import threading
import time
import types
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

from evaluation.censor_mlflow import tags
from guardrails.censor import PROFILE_FILE
from services.llm import MLX_ADAPTERS_DIR, MLX_MODELS_DIR

DATA = Path("data/processed/censor_sft")
EXPERIMENT = "censor-train"
MIN_FREE_PERCENT = 20  # stop training if the whole Mac falls below this much free memory
SMOKE = {
    "iters": 20,
    "steps_per_report": 5,
    "steps_per_eval": 10,
    "val_batches": 5,
    "save_every": 20,
}


def mlx_data(out: Path, data: Path = DATA) -> Path:
    """train.jsonl and valid.jsonl with only the chat messages, as mlx-lm reads them."""
    out.mkdir(parents=True, exist_ok=True)
    for src, dst in (("train", "train"), ("val", "valid")):
        with (
            (data / f"{src}.jsonl").open(encoding="utf-8") as f,
            (out / f"{dst}.jsonl").open("w", encoding="utf-8") as g,
        ):
            for line in f:
                g.write(json.dumps({"messages": json.loads(line)["messages"]}) + "\n")
    return out


def _thinking_off(tokenizer: Any) -> None:
    """mlx-lm renders training chats with thinking on by default; inference (services.llm) turns
    it off. Force it off here too, so training and inference prompts match (for gemma4 thinking
    on adds a flag to the system turn)."""
    render = tokenizer.apply_chat_template

    def apply_chat_template(*a: Any, **k: Any) -> Any:
        return render(*a, **(k | {"enable_thinking": False}))

    tokenizer.apply_chat_template = apply_chat_template


def _system_free_percent() -> int | None:
    """macOS's own free-memory level (0-100), as memory_pressure reports it."""
    try:
        out = subprocess.run(  # noqa: S603  # nosec B603
            ["/usr/sbin/sysctl", "-n", "kern.memorystatus_level"],
            capture_output=True,
            text=True,
            check=True,
            timeout=2,
        )
        return int(out.stdout.strip())
    except (OSError, ValueError, subprocess.SubprocessError):
        return None


def _memory_guard(max_gb: float, min_free_percent: int = MIN_FREE_PERCENT) -> None:
    """Stop the process if MLX memory (active plus cache) passes max_gb, or if the whole Mac drops
    below min_free_percent free. The Mac is shared and has no swap, and a model that does not fit
    (Qwen3.6-27B reached 85-91 GB) must fail cleanly instead of starving other users."""
    import mlx.core as mx

    def stop(reason: str) -> None:
        print(f"memory guard: {reason}, stopping", flush=True)
        os._exit(3)

    def watch() -> None:
        tick = 0
        while True:
            used = (mx.get_active_memory() + mx.get_cache_memory()) / 1e9
            if used > max_gb:
                stop(f"MLX {used:.1f} GB > {max_gb} GB")
            if tick % 10 == 0:  # the system check is slower, once a second
                free = _system_free_percent()
                if free is not None and free < min_free_percent:
                    stop(f"system free memory {free}% < {min_free_percent}%")
            tick += 1
            time.sleep(0.1)

    threading.Thread(target=watch, daemon=True).start()


CKPT = re.compile(r"^(\d{7})_adapters\.safetensors$")  # mlx-lm's checkpoint names
OFFSET_FILE = "resume_offset.txt"


def _prepare_resume(
    adapter_dir: Path, total_iters: int, accumulation: int, schedule: dict[str, Any] | None
) -> tuple[int, Path | None, dict[str, Any] | None]:
    """Continue an interrupted run. mlx-lm numbers checkpoints from 0 in every run, so the last
    run's checkpoints are renamed to global step numbers (global_NNNNNNN_adapters.safetensors)
    first. Returns the global step to continue from, its checkpoint, and the learning-rate
    schedule from that point on (cosine decay continued, no new warm-up)."""
    offset_path = adapter_dir / OFFSET_FILE
    offset = int(offset_path.read_text()) if offset_path.exists() else 0
    for ckpt in adapter_dir.glob("*_adapters.safetensors"):
        if m := CKPT.match(ckpt.name):
            ckpt.rename(adapter_dir / f"global_{offset + int(m.group(1)):07d}_adapters.safetensors")
    done = sorted(adapter_dir.glob("global_*_adapters.safetensors"))
    if not done:
        return 0, None, schedule
    step = int(done[-1].name.split("_")[1])
    offset_path.write_text(str(step))
    if schedule and schedule.get("name") == "cosine_decay":
        init, decay, end = schedule["arguments"]
        updates = min(step // accumulation, decay)
        lr = end + (init - end) * 0.5 * (1 + math.cos(math.pi * updates / decay))
        schedule = {"name": "cosine_decay", "arguments": [lr, max(1, decay - updates), end]}
    return min(step, total_iters), done[-1], schedule


def _reproducible_order(
    original: Callable[..., Iterator[Any]], reseed: Callable[[int], None], seed: int, skip: int
) -> Callable[..., Iterator[Any]]:
    """Wrap mlx-lm's iterate_batches so the training order follows the config's seed. mlx-lm
    shuffles with NumPy's global generator, which train_model never seeds, so every run (and every
    resume) drew a new order. Each epoch is reseeded with seed + epoch just before its shuffle, and
    the first `skip` batches (already trained before a resume) are passed over, so a resumed run
    continues the same order. Validation batches (loop=False) are left as they are."""

    def iterate_batches(*args: Any, loop: bool = False, **kwargs: Any) -> Iterator[Any]:
        if not loop:
            yield from original(*args, **kwargs)
            return
        done = 0
        for epoch in itertools.count():
            reseed(seed + epoch)  # the shuffle is drawn at the first batch, right below
            for batch in original(*args, **kwargs):
                if done >= skip:
                    yield batch
                done += 1

    return iterate_batches


def main() -> None:
    parser = argparse.ArgumentParser(description="Censor LoRA fine-tune (MLX).")
    parser.add_argument("config", type=Path)
    parser.add_argument("--smoke", action="store_true", help="a few steps, to check memory")
    parser.add_argument(
        "--max-memory-gb", type=float, default=60.0, help="abort training above this (shared Mac)"
    )
    parser.add_argument(
        "--cache-limit-gb", type=float, default=4.0, help="cap MLX's cache of reusable buffers"
    )
    parser.add_argument(
        "--resume", action="store_true", help="continue from the adapter's latest checkpoint"
    )
    args = parser.parse_args()
    load_dotenv(".env")

    import mlflow
    from mlx_lm.lora import CONFIG_DEFAULTS, load, load_dataset, train_model
    from mlx_lm.tuner.callbacks import TrainingCallback

    cfg: dict[str, Any] = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    model_name, adapter_name = cfg.pop("model"), cfg.pop("adapter_name")
    data_root = Path(cfg.pop("dataset", str(DATA)))  # e.g. data/processed/censor_sft_v2
    profile = cfg.pop("censor_profile", "rubric")  # how the Censor must prompt this adapter
    if args.smoke:
        cfg |= SMOKE
        adapter_name += "-smoke"
    models_dir = Path(os.environ.get("MLX_MODELS_DIR", MLX_MODELS_DIR))
    adapter_dir = Path(os.environ.get("MLX_ADAPTERS_DIR", MLX_ADAPTERS_DIR)) / adapter_name
    data_dir = mlx_data(Path("data/interim/censor_sft/mlx") / data_root.name, data_root)
    settings = (
        CONFIG_DEFAULTS
        | cfg
        | {
            "model": str(models_dir / model_name),
            "adapter_path": str(adapter_dir),
            "data": str(data_dir),
            "train": True,
            "test": False,
            "report_to": None,
        }
    )
    schedule = settings.get("lr_schedule")
    if schedule and schedule.get("name") == "cosine_decay" and schedule["arguments"][1] == "auto":
        # decay over the whole run: one step per weight update, after the warm-up
        updates = math.ceil(settings["iters"] / settings.get("grad_accumulation_steps", 1))
        decay = max(1, updates - schedule.get("warmup", 0))
        settings["lr_schedule"] = schedule | {
            "arguments": [*schedule["arguments"][:1], decay, *schedule["arguments"][2:]]
        }
    offset, resume_file = 0, None
    if args.resume:
        offset, resume_file, schedule = _prepare_resume(
            adapter_dir,
            settings["iters"],
            settings.get("grad_accumulation_steps", 1),
            settings.get("lr_schedule"),
        )
        if resume_file is not None:
            settings |= {
                "resume_adapter_file": str(resume_file),
                "iters": settings["iters"] - offset,
                "lr_schedule": schedule,
            }
            print(f"resuming from step {offset} ({resume_file.name})", flush=True)
        if settings["iters"] <= 0:
            print("already trained to the last step", flush=True)
            return
    import numpy as np
    from mlx_lm.tuner import trainer

    trainer.iterate_batches = _reproducible_order(
        trainer.iterate_batches, np.random.seed, settings["seed"], offset
    )
    margs = types.SimpleNamespace(**settings)
    manifest = json.loads((data_root / "manifest.json").read_text(encoding="utf-8"))

    class MLflowCallback(TrainingCallback):  # type: ignore[misc]
        def on_train_loss_report(self, info: dict[str, Any]) -> None:
            step = offset + int(info["iteration"])
            mlflow.log_metrics(
                {
                    k: float(info[k])
                    for k in ("train_loss", "learning_rate", "tokens_per_second", "peak_memory")
                    if k in info
                },
                step=step,
            )

        def on_val_loss_report(self, info: dict[str, Any]) -> None:
            mlflow.log_metric(
                "val_loss", float(info["val_loss"]), step=offset + int(info["iteration"])
            )

    mlflow.set_experiment(EXPERIMENT + ("-smoke" if args.smoke else ""))
    with mlflow.start_run(run_name=f"{adapter_name} {model_name}"):
        mlflow.set_tags(
            tags(f"mlx:{model_name}")
            | {
                "adapter": adapter_name,
                "dataset_rubric_sha": manifest["rubric_sha"],
                "smoke": str(args.smoke).lower(),
                "resumed_from_step": str(offset),
            }
        )
        mlflow.log_params(
            {k: str(v) for k, v in cfg.items()}
            | {"base_model": model_name, "dataset": str(data_root)}
        )
        mlflow.log_params({f"data.{k}": v for k, v in manifest["counts"].items()})
        mlflow.log_artifact(str(args.config))
        start = time.monotonic()
        import mlx.core as mx

        mx.set_cache_limit(int(args.cache_limit_gb * 1e9))
        _memory_guard(args.max_memory_gb)
        adapter_dir.mkdir(parents=True, exist_ok=True)
        (adapter_dir / PROFILE_FILE).write_text(profile, encoding="utf-8")
        mlflow.set_tag("censor_profile", profile)
        model, tokenizer = load(margs.model)
        _thinking_off(tokenizer)
        train_set, valid_set, _ = load_dataset(margs, tokenizer)
        train_model(margs, model, train_set, valid_set, MLflowCallback())
        mlflow.log_metric("train_hours", (time.monotonic() - start) / 3600)
        for name in ("adapter_config.json", "adapters.safetensors"):
            path = adapter_dir / name
            if path.exists() and path.stat().st_size < 1_000_000_000:  # keep artifacts small
                mlflow.log_artifact(str(path), artifact_path="adapter")
    print(f"adapter saved to {adapter_dir}")


if __name__ == "__main__":
    main()
