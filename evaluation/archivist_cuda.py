"""Benchmark a CUDA PEFT adapter on the prepared Archivist validation scenes."""

import argparse
import hashlib
import json
from pathlib import Path

from evaluation.archivist import run_case, summarize
from services.extractor.schema import SceneExtraction
from services.llm import parse_json_answer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3.6-27B")
    parser.add_argument("--adapter", type=Path, help="PEFT adapter; omit for the base model")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import torch
    from peft import PeftModel
    from transformers import (
        AutoModelForCausalLM,
        AutoModelForImageTextToText,
        AutoTokenizer,
        BitsAndBytesConfig,
    )

    meta = {
        "model": args.model,
        "adapter": str(args.adapter) if args.adapter else None,
        "dataset_sha256": hashlib.sha256(args.data.read_bytes()).hexdigest(),
        "max_new_tokens": 8192,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    metadata = args.output / "metadata.json"
    if metadata.exists() and json.loads(metadata.read_text()) != meta:
        raise ValueError("benchmark metadata changed")
    metadata.write_text(json.dumps(meta, indent=2) + "\n")
    path = args.output / "results.jsonl"
    done = (
        {json.loads(line)["scene_id"] for line in path.read_text().splitlines()}
        if path.exists()
        else set()
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=False)
    config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model_class = (
        AutoModelForCausalLM
        if "glm-4-9b-chat-hf" in args.model.lower() or "deepseek-r1-distill" in args.model.lower()
        else AutoModelForImageTextToText
    )
    model = model_class.from_pretrained(
        args.model,
        quantization_config=config,
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
        trust_remote_code=False,
    )
    if args.adapter:
        model = PeftModel.from_pretrained(model, str(args.adapter))
        print("adapter reloaded", flush=True)
    model.eval()

    def call(system, user):
        inputs = tokenizer.apply_chat_template(
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
            return_tensors="pt",
            return_dict=True,
        ).to("cuda")
        with torch.inference_mode():
            output = model.generate(**inputs, max_new_tokens=8192, do_sample=False)
        generated = output[0, inputs["input_ids"].shape[-1] :]
        return (
            parse_json_answer(
                tokenizer.decode(generated, skip_special_tokens=True), SceneExtraction
            ),
            inputs["input_ids"].shape[-1],
            len(generated),
        )

    with args.data.open(encoding="utf-8") as source, path.open("a", encoding="utf-8") as out:
        for line in source:
            row = json.loads(line)
            if row["scene_id"] in done:
                continue
            result = run_case(call, row["messages"][0]["content"], row["messages"])
            result |= {"scene_id": row["scene_id"], "book_id": row["book_id"]}
            out.write(json.dumps(result) + "\n")
            out.flush()
            print(row["scene_id"], "accepted=", result["accepted"], flush=True)
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    report = summarize(rows)
    report["books"] = {
        str(book): summarize([r for r in rows if r["book_id"] == book])
        for book in sorted({r["book_id"] for r in rows})
    }
    (args.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
