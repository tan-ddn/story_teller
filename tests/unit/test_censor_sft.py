import json
from pathlib import Path

import pytest

from evaluation.censor_bans import SUITE
from guardrails.censor import CATEGORIES, Scores
from pipelines.censor_sft import build


def scores(**kw: object) -> dict[str, object]:
    base = dict.fromkeys(CATEGORIES, 0) | {"real_person": False, "operational_harm": False}
    return base | kw


def test_norm_hash_ignores_case_whitespace_and_quote_style() -> None:
    assert build.norm_hash("He said “Hello”\n  there") == build.norm_hash('he said "hello" there')
    assert build.norm_hash("a") != build.norm_hash("b")


def test_positives_are_blocked_at_every_rating() -> None:
    assert build._is_positive(scores(sexual=1))
    assert build._is_positive(scores(real_person=True))
    assert build._is_positive(scores(hate=3))
    assert not build._is_positive(scores(violence=2, horror=2))


def test_synthetic_template_fills_share_a_split() -> None:
    val: set[str] = set()
    for t in range(40):
        splits = {
            build._split({"source": "synthetic", "id": f"synth-rp-pos{t:02d}-{k}"}, set())
            for k in range(2)
        }
        assert len(splits) == 1
        val |= splits
    assert val == {"train", "val"}
    assert build._split({"source": "gutenberg", "book": 7}, {7}) == "val"


def test_training_injection_templates_differ_from_the_eval_wrappers() -> None:
    eval_wrappers = json.loads(SUITE.read_text(encoding="utf-8"))["injection_wrappers"]
    for template in build.INJECTION_TEMPLATES:
        sample = template.format(text="X")
        assert all(
            build.norm_hash(sample) != build.norm_hash(w.format(text="X")) for w in eval_wrappers
        )


def test_labels_never_name_an_eval_sample() -> None:
    if not Path("data/processed/scenes.jsonl").exists():
        pytest.skip("needs the local data files")
    eval_ids, _ = build.eval_exclusions()
    labelled = {
        json.loads(line)["id"].split("#")[0]
        for path in build.LABELS.glob("*.jsonl")
        for line in path.open(encoding="utf-8")
    }
    assert not labelled & eval_ids


def test_val_report_counts_errors_as_blocks() -> None:
    from evaluation.censor_sft_val import report

    clean = Scores.model_validate(scores())
    sexual = Scores.model_validate(scores(sexual=1))
    gold = {"a": clean, "b": sexual}
    r = report(gold, {"a": clean, "b": None})
    assert r["error_rate"] == 0.5
    assert r["teen"]["block_recall"] == 1.0  # the failed answer fails closed
    r = report(gold, {"a": sexual, "b": clean})
    assert r["teen"]["block_recall"] == 0.0 and r["teen"]["block_precision"] == 0.0
    assert r["sexual"]["exact"] == 0.0


def test_json_answers_are_parsed_from_model_text() -> None:
    from services.llm import json_system, parse_json_answer

    answer = "<think>\n\n</think>\n\n```json\n" + json.dumps(scores(violence=2)) + "\n```"
    assert parse_json_answer(answer, Scores).violence == 2
    assert "JSON schema" in json_system("RUBRIC", Scores)


def test_mlx_worker_runs_every_job_on_one_thread() -> None:
    import threading
    from concurrent.futures import ThreadPoolExecutor

    from services.llm import _MLXWorker

    worker = _MLXWorker()
    with ThreadPoolExecutor(4) as pool:
        names = set(
            pool.map(lambda _: worker.submit(lambda: threading.current_thread().name), range(8))
        )
    assert names == {"mlx"}
