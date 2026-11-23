import json

import pytest

from training.archivist.cuda_prepare import encode_row, prepare


class Tokenizer:
    def apply_chat_template(self, messages, *, add_generation_prompt, **kwargs):
        assert kwargs["enable_thinking"] is False
        return [1, 2] if add_generation_prompt else [1, 2, 3, 4]


def row(scene, book):
    return {
        "scene_id": scene,
        "book_id": book,
        "messages": [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "scene"},
            {"role": "assistant", "content": "label"},
        ],
    }


def test_mask_and_reject_truncation():
    assert encode_row(Tokenizer(), row("1-1", 1), 4)["labels"] == [-100, -100, 3, 4]
    with pytest.raises(ValueError, match="no truncation"):
        encode_row(Tokenizer(), row("1-1", 1), 3)


def test_prepare_preserves_book_split_and_excludes_test(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    manifest = {
        "counts": {"train": 1, "valid": 1},
        "books": {"train": [1], "valid": [2], "test": [3]},
    }
    (data / "manifest.json").write_text(json.dumps(manifest))
    for split, record in [("train", row("1-1", 1)), ("valid", row("2-1", 2))]:
        (data / f"{split}.jsonl").write_text(json.dumps(record) + "\n")
    out = tmp_path / "out"
    report = prepare(data, out, Tokenizer(), 4)
    assert report["train"]["longest_scene_id"] == "1-1"
    assert not (out / "test.jsonl").exists()
    manifest["books"]["test"] = [1]
    (data / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="leakage"):
        prepare(data, tmp_path / "bad", Tokenizer(), 4)
