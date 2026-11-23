from pipelines.labeling.clean_fable import clean
from services.extractor.schema import SceneExtraction


def test_clean_drops_redundant_modes_and_doubled_links() -> None:
    x = SceneExtraction.model_validate(
        {
            "entities": [
                {
                    "id": "e1",
                    "type": "character",
                    "name": "the white kitten",
                    "known_id": "B12-1",
                    "evidence": "x",
                },
                {
                    "id": "e2",
                    "type": "character",
                    "name": "Snowdrop",
                    "known_id": "B12-1",
                    "evidence": "x",
                },
            ],
            "attributes": [
                {"entity": "e1", "key": "species", "value": "cat", "mode": "main", "evidence": "x"},
                {"entity": "e2", "key": "trait", "value": "shy", "mode": "told", "evidence": "x"},
            ],
            "events": [],
            "relations": [],
            "time": {"mode": "main"},
        }
    )
    registry = {"B12-1": {"name": "the white kitten", "aliases": []}}
    cleaned, modes, links = clean(x, registry)
    assert (modes, links) == (1, 1)
    assert [a.mode for a in cleaned.attributes] == [None, "told"]  # only the redundant one goes
    assert [e.known_id for e in cleaned.entities] == [
        "B12-1",
        None,
    ]  # the registry name match keeps it
