"""CUDA QLoRA training for pre-tokenized full-scene Archivist data."""

import argparse
import json
import math
import re
import time
from pathlib import Path


def validate_smoke(report, audit, model):
    if report.get("model") != model:
        raise ValueError("smoke model does not match full run")
    expected_layers = expected_lora_layers(model)
    if report.get("lora_layers") != expected_layers:
        raise ValueError("smoke did not verify the final 16 language layers")
    for split in ("train", "valid"):
        if report.get(f"{split}_scene") != audit[split]:
            raise ValueError(f"smoke {split} audit does not match full run")
    for group, key in (("train_metrics", "train_loss"), ("eval_metrics", "eval_loss")):
        value = report.get(group, {}).get(key)
        if not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"smoke {key} is missing or non-finite")


def expected_lora_layers(model):
    name = model.lower()
    count = (
        40
        if "glm-4-9b-chat-hf" in name
        else 48
        if "deepseek-r1-distill-qwen-14b" in name
        else 47
        if "glm-4.7-flash" in name
        else 60
        if "gemma-4-31b" in name
        else 64
    )
    return list(range(count - 16, count))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="Qwen/Qwen3.6-27B")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--smoke",
        action="store_true",
        help="four optimizer steps on the longest train and validation scenes",
    )
    mode.add_argument("--full", action="store_true", help="train one epoch on all train scenes")
    parser.add_argument("--smoke-report", type=Path, help="successful matching smoke_metrics.json")
    parser.add_argument("--resume-from-checkpoint", type=Path)
    args = parser.parse_args()
    if args.resume_from_checkpoint:
        if not args.full or args.resume_from_checkpoint.parent.resolve() != args.output.resolve():
            parser.error("resume requires full mode and a checkpoint inside the output directory")
        for name in (
            "adapter_config.json",
            "adapter_model.safetensors",
            "optimizer.pt",
            "scheduler.pt",
            "rng_state.pth",
            "trainer_state.json",
        ):
            if not (args.resume_from_checkpoint / name).is_file():
                raise FileNotFoundError(f"incomplete checkpoint: {name}")
    elif args.output.exists():
        raise FileExistsError(f"output already exists: {args.output}")
    import torch
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import (
        AutoModelForCausalLM,
        AutoModelForImageTextToText,
        AutoTokenizer,
        BitsAndBytesConfig,
        Trainer,
        TrainingArguments,
    )

    if not torch.cuda.is_available():
        raise RuntimeError("a CUDA GPU is required")
    audit = json.loads((args.data / "audit.json").read_text(encoding="utf-8"))
    if audit.get("tokenizer") != args.model:
        raise ValueError("prepared tokenizer and requested model differ")
    if args.full:
        if not args.smoke_report:
            parser.error("--full requires --smoke-report")
        validate_smoke(json.loads(args.smoke_report.read_text(encoding="utf-8")), audit, args.model)
    selected = {}
    all_rows = {}
    for split in ("train", "valid"):
        rows = [
            json.loads(line)
            for line in (args.data / f"{split}.jsonl").read_text(encoding="utf-8").splitlines()
        ]
        if len(rows) != audit[split]["count"]:
            raise ValueError(f"{split}: prepared row count differs from audit")
        matching = [row for row in rows if row["scene_id"] == audit[split]["longest_scene_id"]]
        if len(matching) != 1 or len(matching[0]["input_ids"]) != audit[split]["longest_tokens"]:
            raise ValueError(f"{split}: prepared data does not match audit")
        row = matching[0]
        selected[split] = {"input_ids": row["input_ids"], "labels": row["labels"]}
        all_rows[split] = [{"input_ids": r["input_ids"], "labels": r["labels"]} for r in rows]
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=False)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model_class = (
        AutoModelForCausalLM
        if "glm-4.7-flash" in args.model.lower()
        or "glm-4-9b-chat-hf" in args.model.lower()
        or "deepseek-r1-distill" in args.model.lower()
        else AutoModelForImageTextToText
    )
    model = model_class.from_pretrained(
        args.model,
        quantization_config=quantization,
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
        trust_remote_code=False,
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    wanted = set(expected_lora_layers(args.model))
    targets = [
        name
        for name, module in model.named_modules()
        if (match := re.search(r"(?:^|\.)layers\.(\d+)\.", name))
        and int(match.group(1)) in wanted
        and isinstance(module, torch.nn.Linear)
    ]
    layers = sorted({int(re.search(r"(?:^|\.)layers\.(\d+)\.", name).group(1)) for name in targets})
    if layers != sorted(wanted):
        raise ValueError(f"unexpected LoRA target layers: {layers}")
    model = get_peft_model(
        model,
        LoraConfig(
            r=8, lora_alpha=20, lora_dropout=0, target_modules=targets, task_type="CAUSAL_LM"
        ),
    )
    actual = sorted(
        {
            int(m.group(1))
            for name, param in model.named_parameters()
            if param.requires_grad and (m := re.search(r"(?:^|\.)layers\.(\d+)\.", name))
        }
    )
    unexpected = [
        name
        for name, param in model.named_parameters()
        if param.requires_grad
        and (
            not (match := re.search(r"(?:^|\.)layers\.(\d+)\.", name))
            or int(match.group(1)) not in wanted
        )
    ]
    if actual != layers or unexpected:
        raise ValueError(f"unexpected trainable parameters: layers={actual}, others={unexpected}")
    model.print_trainable_parameters()

    def collate(rows):
        length = max(len(row["input_ids"]) for row in rows)
        return {
            "input_ids": torch.tensor(
                [
                    r["input_ids"] + [tokenizer.pad_token_id] * (length - len(r["input_ids"]))
                    for r in rows
                ]
            ),
            "attention_mask": torch.tensor(
                [[1] * len(r["input_ids"]) + [0] * (length - len(r["input_ids"])) for r in rows]
            ),
            "labels": torch.tensor(
                [r["labels"] + [-100] * (length - len(r["labels"])) for r in rows]
            ),
        }

    settings = TrainingArguments(
        output_dir=str(args.output),
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=4,
        max_steps=4 if args.smoke else -1,
        num_train_epochs=1,
        learning_rate=5e-5,
        bf16=True,
        logging_steps=1 if args.smoke else 5,
        save_steps=4 if args.smoke else 25,
        eval_strategy="no",
        report_to="none",
        remove_unused_columns=False,
        gradient_checkpointing=True,
        seed=0,
    )
    trainer = Trainer(
        model=model,
        args=settings,
        train_dataset=Dataset.from_list(
            [selected["train"]] * 16 if args.smoke else all_rows["train"]
        ),
        eval_dataset=Dataset.from_list([selected["valid"]] if args.smoke else all_rows["valid"]),
        data_collator=collate,
    )
    started = time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    result = trainer.train(
        resume_from_checkpoint=str(args.resume_from_checkpoint)
        if args.resume_from_checkpoint
        else None
    )
    evaluation = trainer.evaluate()
    trainer.save_model(str(args.output / "adapter"))
    report = {
        "model": args.model,
        "lora_layers": layers,
        "train_scene": audit["train"],
        "valid_scene": audit["valid"],
        "elapsed_seconds": time.monotonic() - started,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        "train_metrics": result.metrics,
        "eval_metrics": evaluation,
    }
    report_name = "smoke_metrics.json" if args.smoke else "train_metrics.json"
    (args.output / report_name).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
