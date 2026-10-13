from pipelines.labeling.silver import label_scene
from services.extractor.schema import SceneExtraction, SceneTime

SCENE = "Jim Hawkins stood on the deck."


def extraction(evidence: str) -> SceneExtraction:
    entity = {"id": "e1", "type": "character", "name": "Jim Hawkins", "evidence": evidence}
    return SceneExtraction(
        entities=[entity],  # type: ignore[list-item]
        attributes=[],
        events=[],
        relations=[],
        time=SceneTime(mode="main"),
    )


def test_bad_evidence_is_retried_with_the_quotes_listed() -> None:
    prompts: list[str] = []
    answers = [extraction("Jim stood proudly"), extraction("Jim Hawkins stood")]

    def call(system: str, user: str) -> tuple[SceneExtraction, int, int]:
        prompts.append(user)
        return answers[len(prompts) - 1], 100, 50

    run = label_scene(call, "claude-opus-5", "s1", SCENE)
    assert run.extraction is not None and run.error is None
    assert run.attempts == 2
    assert "- Jim stood proudly" in prompts[1]
    assert (run.input_tokens, run.output_tokens) == (200, 100)
    assert run.cost_usd == round((200 * 5 + 100 * 25) / 1_000_000, 4)


def test_failures_are_recorded_not_raised() -> None:
    def call(system: str, user: str) -> tuple[SceneExtraction, int, int]:
        raise TimeoutError("slow")

    run = label_scene(call, "gpt-6-astra", "s1", SCENE)
    assert run.extraction is None
    assert run.attempts == 2
    assert run.error == "TimeoutError: slow"
