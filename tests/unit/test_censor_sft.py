import json
from pathlib import Path

import pytest

from evaluation.censor_bans import SUITE
from guardrails.censor import CATEGORIES
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
