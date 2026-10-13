from typing import Any

import pytest
from pydantic import ValidationError

from services.extractor.schema import SceneExtraction, check_evidence

SCENE = (
    "Three days later, Jim Hawkins stood on the deck of the Hispaniola.\n"
    "“Captain Smollett,” said Jim, “the cook has a parrot.”"
)


def extraction(**overrides: Any) -> dict[str, Any]:
    data: dict[str, Any] = {
        "entities": [
            {
                "id": "e1",
                "type": "character",
                "name": "Jim Hawkins",
                "aliases": ["Jim"],
                "evidence": "Jim Hawkins stood on the deck",
            },
            {
                "id": "e2",
                "type": "location",
                "name": "the Hispaniola",
                "evidence": "the deck of the Hispaniola",
            },
            {"id": "e3", "type": "character", "name": "the cook", "evidence": "the cook"},
        ],
        "attributes": [
            {"entity": "e3", "key": "owns_pet", "value": "parrot", "evidence": "has a parrot"}
        ],
        "events": [
            {
                "description": "Jim speaks to Captain Smollett about the cook.",
                "participants": ["e1"],
                "location": "e2",
                "evidence": '"Captain Smollett," said Jim',
            }
        ],
        "relations": [],
        "time": {"mode": "main", "relative": "three days later", "evidence": "Three days later"},
    }
    data.update(overrides)
    return data


def test_valid_extraction_and_evidence() -> None:
    parsed = SceneExtraction.model_validate(extraction())
    assert check_evidence(parsed, SCENE) == []  # curly quotes and line breaks are ignored


def test_invented_evidence_is_reported() -> None:
    attrs = [{"entity": "e3", "key": "has_leg", "value": "one", "evidence": "one-legged cook"}]
    parsed = SceneExtraction.model_validate(extraction(attributes=attrs))
    assert check_evidence(parsed, SCENE) == ["one-legged cook"]


@pytest.mark.parametrize(
    "overrides",
    [
        {"relations": [{"src": "e1", "dst": "e9", "type": "knows", "evidence": "x"}]},
        {"attributes": [{"entity": "e1", "key": "Eye Color", "value": "grey", "evidence": "x"}]},
        {
            "events": [
                {"description": "d", "participants": ["e1"], "location": "e3", "evidence": "x"}
            ]
        },
        {"time": {"mode": "daydream"}},
        {"extra_field": 1},
    ],
    ids=["unknown-id", "key-not-snake-case", "location-not-a-place", "bad-mode", "extra-field"],
)
def test_invalid_extractions_are_rejected(overrides: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        SceneExtraction.model_validate(extraction(**overrides))


def test_duplicate_entity_ids_are_rejected() -> None:
    entities = extraction()["entities"]
    with pytest.raises(ValidationError):
        SceneExtraction.model_validate(extraction(entities=[*entities, entities[0]]))


def test_json_schema_exports() -> None:
    schema = SceneExtraction.model_json_schema()
    assert schema["additionalProperties"] is False
    assert set(schema["required"]) == {"entities", "attributes", "events", "relations", "time"}
