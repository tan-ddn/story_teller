from evaluation import censor_bans, censor_fiction
from guardrails.censor import CATEGORIES, Scores


def scored(**kw: object) -> Scores:
    return Scores.model_validate(dict.fromkeys(CATEGORIES, 0) | kw)


def test_bans_report_and_gate() -> None:
    groups = {
        "operational_harm": [("op", "t")],
        "operational_harm_negatives": [("op-n", "t")],
        "real_person": [("rp", "t")],
        "real_person_negatives": [("rp-n", "t")],
        "injection": [("inj", "t")],
    }
    good = {
        "op": scored(operational_harm=True),
        "op-n": scored(),
        "rp": scored(real_person=True),
        "rp-n": scored(),
        "inj": scored(sexual=2),
    }
    r = censor_bans.report(groups, good)  # type: ignore[arg-type]
    assert censor_bans.gate(r) == []
    bad = good | {"op": scored(), "rp-n": scored(real_person=True), "inj": None}
    r = censor_bans.report(groups, bad)
    failures = censor_bans.gate(r)
    assert any("operational_harm recall" in f for f in failures)
    assert any("real_person_negatives" in f for f in failures)
    assert any("classifier errors" in f for f in failures)
    assert r["injection"]["recall"] == 1.0  # an error fails closed, so it still counts as blocked


def test_fiction_gate() -> None:
    ok = {
        "children": {"agreement": 0.8, "false_block": 0.02},
        "teen": {"agreement": 0.95, "false_block": 0.0},
    }
    assert censor_fiction.gate(ok) == []
    worse = ok | {"teen": {"agreement": 0.85, "false_block": 0.05}}
    assert len(censor_fiction.gate(worse)) == 2
