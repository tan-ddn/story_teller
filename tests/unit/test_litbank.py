from pipelines.labeling.litbank import parse_doc
from services.extractor.schema import SceneExtraction, SceneTime, check_evidence

TEXT = "Sir Walter Elliot , of Kellynch Hall , was vain .\nHe loved Kellynch .\nThe ship sailed .\n"
ANN = "\n".join(
    "\t".join(row)
    for row in [
        ["MENTION", "T1", "0", "0", "0", "2", "Sir Walter Elliot", "PER", "PROP"],
        ["MENTION", "T2", "0", "5", "0", "6", "Kellynch Hall", "FAC", "PROP"],
        ["MENTION", "T3", "1", "0", "1", "0", "He", "PER", "PRON"],
        ["MENTION", "T4", "1", "2", "1", "2", "Kellynch", "FAC", "PROP"],
        ["MENTION", "T5", "2", "0", "2", "1", "The ship", "VEH", "NOM"],
        ["COREF", "T3", "Sir_Walter_Elliot-0"],
        ["COREF", "T1", "Sir_Walter_Elliot-0"],
        ["COREF", "T2", "Kellynch_Hall-1"],
        ["COREF", "T4", "Kellynch_Hall-1"],
    ]
)


def test_clusters_become_entities() -> None:
    doc = parse_doc("105_persuasion", ANN, TEXT)
    assert doc.gutenberg_id == 105
    summary = [(e.id, e.type, e.name, e.aliases) for e in doc.entities]
    assert summary == [
        ("e1", "character", "Sir Walter Elliot", []),  # pronoun is not an alias
        ("e2", "location", "Kellynch Hall", ["Kellynch"]),
        ("e3", "object", "The ship", []),  # unclustered mention is a singleton
    ]


def test_entities_pass_evidence_check() -> None:
    doc = parse_doc("105_persuasion", ANN, TEXT)
    extraction = SceneExtraction(
        entities=doc.entities, attributes=[], events=[], relations=[], time=SceneTime(mode="main")
    )
    assert check_evidence(extraction, doc.text) == []
