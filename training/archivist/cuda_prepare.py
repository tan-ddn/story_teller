"""Prepare full-scene CUDA SFT examples without silently truncating labels or prompts."""

import argparse
import hashlib
import json
from pathlib import Path


def encode_row(tokenizer, row, max_length):
    messages = row["messages"]
    if [message["role"] for message in messages] != ["system", "user", "assistant"]:
        raise ValueError(f"{row['scene_id']}: expected system, user, assistant messages")
    options = dict(tokenize=True, enable_thinking=False, return_dict=False)
    prompt = tokenizer.apply_chat_template(messages[:-1], add_generation_prompt=True, **options)
    full = tokenizer.apply_chat_template(messages, add_generation_prompt=False, **options)
    if full[: len(prompt)] != prompt and "deepseek-r1-distill" in tokenizer.name_or_path.lower():
        options["tokenize"] = False
        prefix_text = tokenizer.apply_chat_template(
            messages[:-1], add_generation_prompt=True, **options
        )
        full_text = tokenizer.apply_chat_template(messages, add_generation_prompt=False, **options)
        marker = "<｜Assistant｜>"
        location = full_text.rfind(marker)
        answer_start = location + len(marker)
        if location < 0 or not full_text[answer_start:].startswith(messages[-1]["content"]):
            raise ValueError(f"{row['scene_id']}: unexpected DeepSeek assistant boundary")
        full_text = prefix_text + "</think>\n" + full_text[answer_start:]
        full = tokenizer(full_text, add_special_tokens=False)["input_ids"]
    if full[: len(prompt)] != prompt and "gemma-4-31b" in tokenizer.name_or_path.lower():
        options["tokenize"] = False
        prefix_text = tokenizer.apply_chat_template(
            messages[:-1], add_generation_prompt=True, **options
        )
        full_text = tokenizer.apply_chat_template(messages, add_generation_prompt=False, **options)
        marker = "<|turn>model\n"
        location = full_text.rfind(marker)
        if location < 0 or not prefix_text.startswith(full_text[:location] + marker):
            raise ValueError(f"{row['scene_id']}: unexpected Gemma assistant boundary")
        answer_start = location + len(marker)
        if not full_text[answer_start:].startswith(messages[-1]["content"]):
            raise ValueError(f"{row['scene_id']}: Gemma answer differs from target")
        full_text = prefix_text + full_text[answer_start:]
        full = tokenizer(full_text, add_special_tokens=False)["input_ids"]
    if full[: len(prompt)] != prompt or len(prompt) >= len(full):
        raise ValueError(f"{row['scene_id']}: assistant boundary does not match full chat")
    if len(full) > max_length:
        raise ValueError(
            f"{row['scene_id']}: {len(full)} tokens exceeds {max_length}; no truncation"
        )
    return {"input_ids": full, "labels": [-100] * len(prompt) + full[len(prompt) :]}


def prepare(data, output, tokenizer, max_length):
    manifest = json.loads((data / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("diagnostic_only"):
        raise ValueError("diagnostic-only window data is not suitable for training")
    books, ids, encoded = {}, set(), {}
    reports = {"tokenizer": getattr(tokenizer, "name_or_path", None)}
    for split in ("train", "valid"):
        path = data / f"{split}.jsonl"
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        if len(rows) != manifest["counts"][split]:
            raise ValueError(f"{split}: count differs from manifest")
        books[split] = {row["book_id"] for row in rows}
        if books[split] != set(manifest["books"][split]):
            raise ValueError(f"{split}: books differ from manifest")
        examples = []
        for row in rows:
            if row["scene_id"] in ids:
                raise ValueError(f"duplicate scene {row['scene_id']}")
            ids.add(row["scene_id"])
            item = encode_row(tokenizer, row, max_length)
            examples.append((row["scene_id"], item))
        longest = max(examples, key=lambda item: len(item[1]["input_ids"]))
        encoded[split] = examples
        reports[split] = {
            "count": len(rows),
            "longest_scene_id": longest[0],
            "longest_tokens": len(longest[1]["input_ids"]),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    training_books = books["train"] | books["valid"]
    if books["train"] & books["valid"] or training_books & set(manifest["books"]["test"]):
        raise ValueError("book leakage between splits")
    output.mkdir(parents=True, exist_ok=False)
    for split, examples in encoded.items():
        with (output / f"{split}.jsonl").open("w", encoding="utf-8") as f:
            for scene_id, item in examples:
                f.write(json.dumps({"scene_id": scene_id, **item}) + "\n")
    (output / "audit.json").write_text(json.dumps(reports, indent=2) + "\n", encoding="utf-8")
    return reports


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/interim/archivist_sft"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="Qwen/Qwen3.6-27B")
    parser.add_argument("--max-length", type=int, default=10240)
    args = parser.parse_args()
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=False)
    print(json.dumps(prepare(args.data, args.output, tokenizer, args.max_length), indent=2))


if __name__ == "__main__":
    main()
