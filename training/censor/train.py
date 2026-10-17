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
import json
import os
import threading
import time
import types
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

from evaluation.censor_mlflow import tags
from services.llm import MLX_ADAPTERS_DIR, MLX_MODELS_DIR

DATA = Path("data/processed/censor_sft")
EXPERIMENT = "censor-train"
SMOKE = {
    "iters": 20,
    "steps_per_report": 5,
    "steps_per_eval": 10,
    "val_batches": 5,
    "save_every": 20,
}


def mlx_data(out: Path) -> Path:
    """train.jsonl and valid.jsonl with only the chat messages, as mlx-lm reads them."""
    out.mkdir(parents=True, exist_ok=True)
    for src, dst in (("train", "train"), ("val", "valid")):
        with (
            (DATA / f"{src}.jsonl").open(encoding="utf-8") as f,
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


def _memory_guard(max_gb: float) -> None:
    """Stop the process if MLX memory passes max_gb. The Mac is shared, and a model that does not
    fit (Qwen3.6-27B reached 85-91 GB) must fail cleanly instead of starving other users."""
    import mlx.core as mx

    def watch() -> None:
        while True:
            used = mx.get_active_memory() / 1e9
            if used > max_gb:
                print(f"memory guard: {used:.1f} GB > {max_gb} GB, stopping", flush=True)
                os._exit(3)
            time.sleep(0.5)

    threading.Thread(target=watch, daemon=True).start()


def main() -> None:
    parser = argparse.ArgumentParser(description="Censor LoRA fine-tune (MLX).")
    parser.add_argument("config", type=Path)
    parser.add_argument("--smoke", action="store_true", help="a few steps, to check memory")
    parser.add_argument(
        "--max-memory-gb", type=float, default=60.0, help="abort training above this (shared Mac)"
    )
    args = parser.parse_args()
    load_dotenv(".env")

    import mlflow
    from mlx_lm.lora import CONFIG_DEFAULTS, load, load_dataset, train_model
    from mlx_lm.tuner.callbacks import TrainingCallback

    cfg: dict[str, Any] = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    model_name, adapter_name = cfg.pop("model"), cfg.pop("adapter_name")
    if args.smoke:
        cfg |= SMOKE
        adapter_name += "-smoke"
    models_dir = Path(os.environ.get("MLX_MODELS_DIR", MLX_MODELS_DIR))
    adapter_dir = Path(os.environ.get("MLX_ADAPTERS_DIR", MLX_ADAPTERS_DIR)) / adapter_name
    data_dir = mlx_data(Path("data/interim/censor_sft/mlx"))
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
    margs = types.SimpleNamespace(**settings)
    manifest = json.loads((DATA / "manifest.json").read_text(encoding="utf-8"))

    class MLflowCallback(TrainingCallback):  # type: ignore[misc]
        def on_train_loss_report(self, info: dict[str, Any]) -> None:
            step = int(info["iteration"])
            mlflow.log_metrics(
                {
                    k: float(info[k])
                    for k in ("train_loss", "learning_rate", "tokens_per_second", "peak_memory")
                    if k in info
                },
                step=step,
            )

        def on_val_loss_report(self, info: dict[str, Any]) -> None:
            mlflow.log_metric("val_loss", float(info["val_loss"]), step=int(info["iteration"]))

    mlflow.set_experiment(EXPERIMENT + ("-smoke" if args.smoke else ""))
    with mlflow.start_run(run_name=f"{adapter_name} {model_name}"):
        mlflow.set_tags(
            tags(f"mlx:{model_name}")
            | {
                "adapter": adapter_name,
                "dataset_rubric_sha": manifest["rubric_sha"],
                "smoke": str(args.smoke).lower(),
            }
        )
        mlflow.log_params({k: str(v) for k, v in cfg.items()} | {"base_model": model_name})
        mlflow.log_params({f"data.{k}": v for k, v in manifest["counts"].items()})
        mlflow.log_artifact(str(args.config))
        start = time.monotonic()
        _memory_guard(args.max_memory_gb)
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
