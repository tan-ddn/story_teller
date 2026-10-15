import json
from pathlib import Path

from evaluation.censor import (
    BOUNDARY,
    LABELS,
    RELABEL,
    Sample,
    classify_all,
    expected_block,
    gate,
    is_clean,
    is_hard_ban,
    report,
    sample,
    should_block,
)
from guardrails.censor import CATEGORIES, Censor, Scores

CLEAN = dict.fromkeys(LABELS, 0)


def s(sid: str, **labels: int) -> Sample:
    return Sample(sid, "text", CLEAN | labels)


def scored(**kw: int) -> Scores:
    return Scores.model_validate(dict.fromkeys(CATEGORIES, 0) | kw)


def test_should_block_follows_the_teen_policy() -> None:
    assert should_block(s("a", S=1)) and should_block(s("b", H2=1)) and should_block(s("c", V2=1))
    assert not should_block(s("v", V=1))  # violence 2 is within teen
    assert not should_block(s("d"))
    assert is_clean(s("d")) and not is_clean(Sample("e", "t", {"S": 0}))


def test_report_and_gate() -> None:
    bench = [s("pos", S=1), s("clean")]
    boundary = [{"id": "b", "text": "t", "expected": {"violence": 2}}]
    good: dict[str, Scores | None] = {
        "pos": scored(sexual=1),
        "clean": scored(),
        "b": scored(violence=2),
    }
    r = report(bench, boundary, good)
    assert (r["block_recall"], r["false_block"], r["boundary_agreement"]) == (1.0, 0.0, 1.0)
    assert r["detection"]["S"] == {"n": 1, "rate": 1.0}
    assert gate(r) == []

    bad: dict[str, Scores | None] = {"pos": scored(), "clean": None, "b": scored()}
    r = report(bench, boundary, bad)
    assert r["block_recall"] == 0.0 and r["false_block"] == 1.0  # errors count as blocks
    assert len(gate(r)) == 2  # boundary agreement and errors; no hard-ban samples here


def test_sample_is_deterministic_and_half_clean() -> None:
    bench = [s(f"p{i}", V=1) for i in range(50)] + [s(f"c{i}") for i in range(50)]
    a, b = sample(bench, 20), sample(bench, 20)
    assert [x.id for x in a] == [x.id for x in b]
    assert sum(is_clean(x) for x in a) == 10


def test_boundary_set_is_well_formed() -> None:
    rows = [json.loads(line) for line in Path(BOUNDARY).open(encoding="utf-8")]
    assert len({r["id"] for r in rows}) == len(rows)
    for r in rows:
        assert set(r["expected"]) <= set(CATEGORIES) - {"sexual"}
        scored(**r["expected"])  # valid scores


def test_scores_are_cached_per_rubric_and_text(tmp_path: Path) -> None:
    calls: list[str] = []

    def classify(system: str, user: str) -> tuple[Scores, int, int]:
        calls.append(user)
        if "broken" in user:
            raise TimeoutError
        return scored(violence=1), 1000, 1000

    censor = Censor(classify)
    cache = tmp_path / "scores.jsonl"
    first, spent = classify_all(censor, "o4-mini", {"a": "one", "x": "broken"}, 2, cache)
    assert first["a"] == scored(violence=1) and first["x"] is None
    assert spent["new_calls"] == 1 and spent["new_usd"] == round(
        (1000 * 1.10 + 1000 * 4.40) / 1e6, 4
    )

    calls.clear()
    texts = {"a": "one", "b": "two", "x": "broken"}
    second, spent = classify_all(censor, "o4-mini", texts, 2, cache)
    assert sorted(calls) == ["<text>\nbroken\n</text>", "<text>\ntwo\n</text>"]  # a is cached
    assert spent["cached_usd"] > 0

    calls.clear()
    censor.rubric += "\nchanged"
    classify_all(censor, "o4-mini", {"a": "one"}, 1, cache)
    assert calls == ["<text>\none\n</text>"]  # a new rubric invalidates the cache


def test_relabels_override_the_benchmark_clean_label() -> None:
    rows = [json.loads(line) for line in Path(RELABEL).open(encoding="utf-8")]
    assert {r["teen"] for r in rows} <= {"allow", "block", "unsure"}
    assert all("text" not in r for r in rows)  # no benchmark text is committed
    by_label = {r["teen"]: r["id"] for r in rows}
    assert expected_block(s(by_label["block"])) is True
    assert expected_block(s(by_label["allow"])) is False
    assert expected_block(s(by_label["unsure"])) is None


def test_hard_ban_samples_are_the_sexual_blocks() -> None:
    rows = [json.loads(line) for line in Path(RELABEL).open(encoding="utf-8")]
    sexual = next(r["id"] for r in rows if r["teen"] == "block" and r["why"].startswith("sexual"))
    hate = next(r["id"] for r in rows if r["teen"] == "block" and r["why"].startswith("hate"))
    assert is_hard_ban(s(sexual)) and not is_hard_ban(s(hate))
