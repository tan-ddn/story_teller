"""Censor: family-friendly rating checks for generated scenes and uploaded manuscripts (§11).

A classifier scores the text 0-3 per category (guardrails/rubric.md). Rules then compare the
scores with the versioned policy for the project rating (guardrails/policies/*.yaml) and the hard
bans. Censor fails closed: if the classifier errors, the text is not allowed.
"""

from collections.abc import Callable
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

HERE = Path(__file__).parent
DEFAULT_MODEL = "qwen3.6:27b"  # local, via Ollama; see services/llm.py
CATEGORIES = ("violence", "sexual", "profanity", "substance", "self_harm", "horror", "hate")
Rating = Literal["children", "teen"]
Score = Field(ge=0, le=3)


Category = Literal["violence", "sexual", "profanity", "substance", "self_harm", "horror", "hate"]


class Quote(BaseModel):
    model_config = ConfigDict(extra="forbid")

    category: Category
    text: str  # a few words from the text that show the category


class Scores(BaseModel):
    """Classifier output."""

    model_config = ConfigDict(extra="forbid")

    violence: int = Score
    sexual: int = Score
    profanity: int = Score
    substance: int = Score
    self_harm: int = Score
    horror: int = Score
    hate: int = Score
    real_person: bool = False
    operational_harm: bool = False
    quotes: list[Quote] = []  # a list, not a dict: OpenAI strict schemas reject free-form keys


Classify = Callable[[str, str], tuple[Scores, int, int]]  # services.llm.parse_call(model, Scores)


@dataclass(frozen=True)
class Warning:
    """A score above the rating's usual level but within its block level."""

    category: str
    score: int
    message: str  # plain sentences for the author


@dataclass(frozen=True)
class Verdict:
    allowed: bool
    reasons: tuple[str, ...] = ()  # why it was blocked, as plain sentences
    hard_ban: bool = False  # never allowed at any rating; the text must not be stored
    unavailable: bool = False  # the classifier failed; fail closed, do not try to edit around it
    scores: Scores | None = None
    policy_version: int = 0
    warnings: tuple[Warning, ...] = ()
    labels: tuple[str, ...] = ()  # short content warnings for chapters and exports


@dataclass(frozen=True)
class Policy:
    rating: Rating
    version: int
    warn: dict[str, int]  # scores above this get a warning tag
    block: dict[str, int]  # scores above this are rejected


@cache
def policy(rating: Rating) -> Policy:
    data = yaml.safe_load((HERE / "policies" / f"{rating}.yaml").read_text(encoding="utf-8"))
    for key in ("warn", "block"):
        if set(data[key]) != set(CATEGORIES):
            raise ValueError(f"{rating} policy {key} must set exactly {CATEGORIES}")
    if any(data["warn"][c] > data["block"][c] for c in CATEGORIES):
        raise ValueError(f"{rating} policy: warn levels must not exceed block levels")
    return Policy(data["rating"], data["version"], data["warn"], data["block"])


# What each score means, for the author. Category -> score -> sentence.
MEANING = {
    "violence": {
        1: "It has mild peril, such as threats, scuffles or minor injuries.",
        2: "It has fights or battles in which characters are hurt or killed.",
        3: "It has graphic violence, with detailed injuries, gore or cruelty.",
    },
    "sexual": {
        1: "It has sexual innuendo or suggestive description.",
        2: "It describes or implies a sexual situation.",
        3: "It has explicit sexual content.",
    },
    "profanity": {
        1: 'Some characters use mild swear words, such as "damn" or "hell".',
        2: "Characters swear often or use strong swear words.",
        3: "It has slurs or a lot of strong swearing.",
    },
    "substance": {
        1: "An adult is shown tipsy or drunk in a light way, or drugs are mentioned.",
        2: "Characters are shown drunk or on drugs, or a young character drinks, smokes or "
        "uses drugs.",
        3: "Drug use is made to look attractive or is described step by step.",
    },
    "self_harm": {
        1: "A character feels deep sadness or wishes they could disappear.",
        2: "It mentions or shows self-harm or suicide.",
        3: "It describes how someone harms themselves, or presents self-harm as a solution.",
    },
    "horror": {
        1: "It has spooky moments or a scary creature.",
        2: "It has frightening scenes, such as menace, disturbing images or the death of a "
        "loved character.",
        3: "It has intense horror, such as sustained terror or body horror.",
    },
    "hate": {
        1: "It shows prejudice against a group, and the story treats it as wrong.",
        2: "Characters use slurs or demeaning stereotypes.",
        3: "It presents hatred of a group as acceptable.",
    },
}
AUDIENCE = {"children": "children (12 and under)", "teen": "teenagers (13 to 17)"}

# Short content warnings shown on chapters and exports, by category and minimum score.
LABELS = {
    "violence": {1: "mild peril", 2: "violence"},
    "profanity": {1: "mild language"},
    "substance": {1: "alcohol", 2: "substance use"},
    "self_harm": {1: "themes of sadness", 2: "self-harm themes"},
    "horror": {1: "scary moments", 2: "frightening scenes"},
    "hate": {1: "depictions of prejudice"},
}


def content_labels(scores: Scores) -> tuple[str, ...]:
    found = []
    for category, levels in LABELS.items():
        score = getattr(scores, category)
        hits = [label for level, label in levels.items() if score >= level]
        if hits:
            found.append(hits[-1])
    return tuple(found)


def _explain(scores: Scores, category: str) -> str:
    message = MEANING[category][getattr(scores, category)]
    if quote := next((q.text for q in scores.quotes if q.category == category), None):
        message += f' For example: "{quote}".'
    return message


def judge(scores: Scores, rating: Rating) -> Verdict:
    """Apply the rules to classifier scores. Pure, so it is unit-tested without a model."""
    p = policy(rating)
    audience = AUDIENCE[rating]
    bans = []
    if scores.sexual > 0:
        bans.append(f"Sexual content is never allowed. {_explain(scores, 'sexual')}")
    if scores.real_person:
        bans.append("Demeaning or defamatory content about a real private person is never allowed.")
    if scores.operational_harm:
        bans.append(
            "Real-world instructions that could cause harm are never allowed, even in fiction."
        )
    blocked, warnings = [], []
    for category in CATEGORIES:
        score = getattr(scores, category)
        if category == "sexual" or score == 0:
            continue  # sexual content is handled as a hard ban above
        if score > p.block[category]:
            blocked.append(f"This is too strong for {audience}. {_explain(scores, category)}")
        elif score > p.warn[category]:
            message = f"This is stronger than usual for {audience}. {_explain(scores, category)}"
            warnings.append(Warning(category, score, message))
    return Verdict(
        allowed=not bans and not blocked,
        reasons=tuple(bans + blocked),
        hard_ban=bool(bans),
        scores=scores,
        policy_version=p.version,
        warnings=tuple(warnings),
        labels=content_labels(scores),
    )


RETRY_NOTE = (
    "Your previous answer was not a valid JSON object for the required schema. Answer again with "
    "only the JSON object: the seven category scores as integers 0-3, real_person, "
    "operational_harm and quotes."
)


class Censor:
    def __init__(self, classify: Classify) -> None:
        self.classify = classify
        self.rubric = (HERE / "rubric.md").read_text(encoding="utf-8")

    def classify_text(self, text: str) -> tuple[Scores, int, int]:
        """Scores plus input and output tokens.

        An answer that is not valid Scores JSON (a ValueError, which includes pydantic's
        ValidationError) gets one retry with a correction appended, so a deterministic model does
        not repeat the same output. A second bad answer raises, and check() fails closed. Other
        errors (network, server) are not retried here."""
        user = f"<text>\n{text}\n</text>"
        try:
            return self.classify(self.rubric, user)
        except ValueError as first:
            retry = f"{user}\n\n{RETRY_NOTE}"
            try:
                scores, tokens_in, tokens_out = self.classify(self.rubric, retry)
            except ValueError:
                raise first from None
            return scores, tokens_in, tokens_out

    def scores(self, text: str) -> Scores:
        return self.classify_text(text)[0]

    def check(self, text: str, rating: Rating) -> Verdict:
        try:
            scores = self.scores(text)
        except Exception as e:  # noqa: BLE001 - fail closed on any classifier failure
            return Verdict(False, (f"censor unavailable: {type(e).__name__}",), unavailable=True)
        return judge(scores, rating)


UploadAction = Literal["accept", "flag", "reject"]


def check_upload(censor: Censor, text: str, rating: Rating) -> tuple[UploadAction, Verdict]:
    """Uploaded manuscripts: within the project rating -> accept; over it but within teen ->
    flag for the author; over teen, a hard ban or no verdict -> reject (the text is not stored).
    One classifier call, judged against both teen and the project rating."""
    try:
        scores = censor.scores(text)
    except Exception as e:  # noqa: BLE001 - fail closed
        down = Verdict(False, (f"censor unavailable: {type(e).__name__}",), unavailable=True)
        return "reject", down
    teen, project = judge(scores, "teen"), judge(scores, rating)
    if not teen.allowed:
        return "reject", teen
    return ("accept" if project.allowed else "flag"), project
