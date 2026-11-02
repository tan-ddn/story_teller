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


def test_mlx_worker_stops_on_a_stuck_call(monkeypatch: pytest.MonkeyPatch) -> None:
    import threading

    from services import llm

    def exit_(code: int) -> None:
        raise SystemExit(code)

    monkeypatch.setattr("services.llm.os._exit", exit_)
    release = threading.Event()
    worker = llm._MLXWorker()
    with pytest.raises(SystemExit) as stopped:
        worker.submit(release.wait, timeout=0.05)
    release.set()
    assert stopped.value.code == 4


def test_epoch_checkpoints_are_split_and_ranked(tmp_path: Path) -> None:
    from training.censor import checkpoints

    run = tmp_path / "run"
    run.mkdir()
    (run / "adapter_config.json").write_text("{}")
    for step in (1200, 2400, 4800):
        (run / f"{step:07d}_adapters.safetensors").write_bytes(b"w")
    made = checkpoints.split(run, every=2400)
    assert [p.name for p in made] == ["run-epoch1", "run-epoch2"]
    assert (made[0] / "adapters.safetensors").read_bytes() == b"w"
    good = {
        "children": {"agreement": 0.9},
        "teen": {"agreement": 0.9, "block_recall": 0.9, "block_precision": 0.9},
    }
    worse = {
        "children": {"agreement": 0.8},
        "teen": {"agreement": 0.9, "block_recall": 1.0, "block_precision": 1.0},
    }
    assert checkpoints.score(good) > checkpoints.score(worse)


def test_plan_config_derives_epochs_and_schedule(tmp_path: Path) -> None:
    import yaml

    from training.censor import plan

    data = tmp_path / "data"
    data.mkdir()
    (data / "train.jsonl").write_text("{}\n" * 100)
    base = tmp_path / "base.yaml"
    base.write_text(
        yaml.safe_dump(
            {
                "model": "M",
                "adapter_name": "a",
                "batch_size": 1,
                "grad_accumulation_steps": 4,
                "learning_rate": 5e-5,
            }
        )
    )
    cfg = plan.config(base, data, 3, "new", tmp_path / "out.yaml")
    assert (cfg["iters"], cfg["save_every"]) == (300, 100)
    assert cfg["lr_schedule"]["arguments"] == [5e-5, 75, 1.0e-6]
    assert cfg["adapter_name"] == "new" and cfg["dataset"] == str(data)


def test_resume_renames_checkpoints_and_continues_the_schedule(tmp_path: Path) -> None:
    from training.censor import train

    run = tmp_path / "run"
    run.mkdir()
    for step in (400, 800):
        (run / f"{step:07d}_adapters.safetensors").write_bytes(b"w")
    sched = {"name": "cosine_decay", "warmup": 30, "arguments": [5e-5, 500, 1e-6]}
    step, ckpt, new = train._prepare_resume(run, 2000, 4, sched)
    assert step == 800 and ckpt is not None and ckpt.name == "global_0000800_adapters.safetensors"
    assert new is not None and new["arguments"][1] == 300 and "warmup" not in new
    assert 1e-6 < new["arguments"][0] < 5e-5
    # a second interrupted segment numbered from 0 again continues from the global step
    (run / "0000400_adapters.safetensors").write_bytes(b"w")
    step, ckpt, _ = train._prepare_resume(run, 2000, 4, sched)
    assert step == 1200 and ckpt is not None and ckpt.name == "global_0001200_adapters.safetensors"
