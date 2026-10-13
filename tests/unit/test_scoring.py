from typing import Any

from evaluation.extraction import align_entities, score
from services.extractor.schema import SceneExtraction


def extraction(entities: list[dict[str, Any]], **rest: Any) -> SceneExtraction:
    data: dict[str, Any] = {
        "entities": [{"evidence": "x", **e} for e in entities],
        "attributes": [],
        "events": [],
        "relations": [],
        "time": {"mode": "main"},
    }
    data.update(rest)
    return SceneExtraction.model_validate(data)


GOLD = extraction(
    [
        {"id": "e1", "type": "character", "name": "Jim Hawkins", "aliases": ["Jim"]},
        {"id": "e2", "type": "vehicle", "name": "the Hispaniola"},
        {"id": "e3", "type": "character", "name": "Long John Silver"},
    ],
    relations=[{"src": "e1", "dst": "e3", "type": "friend_of", "evidence": "x"}],
    attributes=[{"entity": "e3", "key": "occupation", "value": "cook", "evidence": "x"}],
    events=[{"description": "d", "participants": ["e1"], "evidence": "Jim boarded the ship"}],
    time={"mode": "main", "cues": [{"kind": "relative", "value": "Next day", "evidence": "x"}]},
)


def test_alignment_uses_aliases_token_subsets_and_ignores_articles() -> None:
    pred = extraction(
        [
            {"id": "e1", "type": "vehicle", "name": "Hispaniola"},
            {"id": "e2", "type": "character", "name": "Jim"},
            {"id": "e3", "type": "character", "name": "Silver"},
            {"id": "e4", "type": "character", "name": "Ben Gunn"},
        ]
    )
    assert align_entities(pred.entities, GOLD.entities) == {"e1": "e2", "e2": "e1", "e3": "e3"}


def test_relations_and_attributes_are_scored_through_alignment() -> None:
    pred = extraction(
        [
            {"id": "e1", "type": "character", "name": "Silver"},
            {"id": "e2", "type": "character", "name": "Jim"},
        ],
        # friend_of is symmetric, so the reversed direction still matches
        relations=[{"src": "e1", "dst": "e2", "type": "friend_of", "evidence": "x"}],
        attributes=[
            {"entity": "e1", "key": "occupation", "value": "ship's cook", "evidence": "x"},
            {"entity": "e2", "key": "age", "value": "young", "evidence": "x"},
        ],
        events=[{"description": "d", "participants": ["e2"], "evidence": "Jim boarded the ship"}],
        time={"mode": "main", "cues": [{"kind": "relative", "value": "next day", "evidence": "x"}]},
    )
    s = score(pred, GOLD)
    assert (s["entities"].tp, s["entities"].pred, s["entities"].gold) == (2, 2, 3)
    assert s["relations"].tp == 1
    assert (s["attributes"].tp, s["attributes"].pred) == (1, 2)
    assert s["events"].tp == 1
    assert s["time_cues"].tp == 1
    assert round(s["entities"].f1, 3) == 0.8
