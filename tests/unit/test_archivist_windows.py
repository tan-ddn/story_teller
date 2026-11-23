from pipelines.archivist_sft.build import prompt
from pipelines.archivist_sft.windows import window_records
from services.extractor.schema import SceneExtraction, check_evidence


def test_windows_report_fact_whose_entity_evidence_is_out_of_range():
    text = "Alice entered. " + "x" * 80 + " Alice left."
    target = SceneExtraction.model_validate(
        {
            "entities": [
                {"id": "e1", "type": "character", "name": "Alice", "evidence": "Alice entered."}
            ],
            "attributes": [],
            "events": [
                {"description": "Alice left", "participants": ["e1"], "evidence": "Alice left."}
            ],
            "relations": [],
            "time": {"mode": "main"},
        }
    )
    row = {
        "scene_id": "11-c001-s01",
        "book_id": 11,
        "messages": [
            {"role": "system", "content": "rules"},
            {"role": "user", "content": prompt(text)},
            {"role": "assistant", "content": target.model_dump_json()},
        ],
    }
    windows, coverage = window_records(row, size=45, overlap=10)
    assert coverage["entities"]["covered"] == 1
    assert coverage["events"]["covered"] == 0
    assert coverage["events"]["uncovered_indexes"] == [0]
    for window in windows:
        extraction = SceneExtraction.model_validate_json(window["messages"][2]["content"])
        fragment = (
            window["messages"][1]["content"].split("\n<scene>\n", 1)[1].rsplit("\n</scene>", 1)[0]
        )
        assert not check_evidence(extraction, fragment)
