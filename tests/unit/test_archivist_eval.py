import json

from evaluation.archivist import run_case, summarize
from services.extractor.schema import SceneExtraction


def test_failed_extraction_counts_as_missed_facts():
    gold = {
        "entities": [
            {"id": "e1", "type": "character", "name": "Alice", "evidence": "Alice walked"}
        ],
        "attributes": [],
        "events": [],
        "relations": [],
        "time": {"mode": "main"},
    }
    messages = [
        {"role": "system", "content": ""},
        {"role": "user", "content": "\n<scene>\nAlice walked.\n</scene>"},
        {"role": "assistant", "content": json.dumps(gold)},
    ]

    def bad_call(system, user):
        raise ValueError("bad JSON")

    result = run_case(bad_call, "", messages)
    assert not result["accepted"]
    summary = summarize([result])
    assert summary["scores"]["entities"]["gold"] == 1
    assert summary["scores"]["entities"]["recall"] == 0
    assert summary["attempt_statuses"]["invalid_schema"] == 2


def test_retry_after_bad_evidence():
    data = {
        "entities": [
            {"id": "e1", "type": "character", "name": "Alice", "evidence": "Alice walked"}
        ],
        "attributes": [],
        "events": [],
        "relations": [],
        "time": {"mode": "main"},
    }
    valid = SceneExtraction.model_validate(data)
    bad = valid.model_copy(deep=True)
    bad.entities[0].evidence = "Alice ran"
    answers = iter([bad, valid])
    messages = [
        {"role": "system", "content": ""},
        {"role": "user", "content": "\n<scene>\nAlice walked.\n</scene>"},
        {"role": "assistant", "content": valid.model_dump_json()},
    ]
    result = run_case(lambda system, user: (next(answers), 5, 5), "", messages)
    assert result["accepted"]
    assert [a["status"] for a in result["attempts"]] == ["evidence_mismatch", "valid"]
    assert result["scores"]["entities"]["tp"] == 1
