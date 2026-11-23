import json
from pathlib import Path

from evaluation.censor_window_compare import read_baseline
from guardrails.censor import DEFAULT_MODEL


def test_baseline_cache_requires_matching_rubric_and_keeps_text_hash(tmp_path: Path) -> None:
    eval_dir = tmp_path / "censor_eval"
    fiction_dir = tmp_path / "censor_fiction"
    eval_dir.mkdir()
    fiction_dir.mkdir()
    scores = {
        "violence": 0,
        "sexual": 0,
        "profanity": 0,
        "substance": 0,
        "self_harm": 0,
        "horror": 0,
        "hate": 0,
    }
    rows = [
        {"id": "one", "text_sha": "text-a", "rubric_sha": "right", "scores": scores},
        {"id": "two", "text_sha": "text-b", "rubric_sha": "wrong", "scores": scores},
    ]
    (eval_dir / f"{DEFAULT_MODEL}.scores.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n"
    )
    (fiction_dir / f"censor.{DEFAULT_MODEL}.jsonl").write_text(
        json.dumps({"id": "three", "text_sha": "text-c", "prompt_sha": "right", "result": scores})
        + "\n"
    )
    result = read_baseline(tmp_path, "right")
    assert set(result) == {("one", "text-a"), ("three", "text-c")}
