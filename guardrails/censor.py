"""Censor: family-friendly rating checks for generated scenes and uploaded manuscripts (§11).

A classifier scores the text 0-3 per category (guardrails/rubric.md). Rules then compare the
scores with the versioned policy for the project rating (guardrails/policies/*.yaml) and the hard
bans. Censor fails closed: if the classifier errors, the text is not allowed.
"""

import re
from collections.abc import Callable
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator

HERE = Path(__file__).parent
# Fine-tuned gemma-4-26b (LoRA, dataset v3), the only Censor that passes every gate; see
# docs/censor-finetune.md. Runs with MLX on Apple Silicon: the adapter must be in MLX_ADAPTERS_DIR
# and the base model in MLX_MODELS_DIR (services/llm.py).
DEFAULT_MODEL = "mlx:gemma-4-26b-a4b-it-4bit+censor-lora-v3-gemma4_26b_a4b"
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

    @field_validator("quotes", mode="before")
    @classmethod
    def _drop_ban_quotes(cls, value: Any) -> Any:
        """Models often quote evidence for a ban too ("operational_harm", "real_person"). Quotes
        exist only for the seven score categories, so such quotes are dropped instead of
        rejecting the whole answer, which hid the very ban decisions that matter."""
        if isinstance(value, list):
            return [q for q in value if not isinstance(q, dict) or q.get("category") in CATEGORIES]
        return value


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


# Censor profiles. "rubric": the rubric is the system prompt and a scene is rated in one call.
# "short": for models fine-tuned without the rubric in the prompt (Qwen3.6 on the Mac only trains
# on short sequences): a one-line system prompt, and texts are rated in chunks of at most
# SHORT_CHUNK_CHARS (~512 tokens), the size they were trained on, then combined.
SHORT_SYSTEM = (
    "You are Censor, the content rater for a family-friendly fiction platform. Rate the text."
)
SHORT_CHUNK_CHARS = 2000
PROFILE_FILE = "censor_profile.txt"  # written into a LoRA adapter folder by training


def split_chunks(text: str, max_chars: int) -> list[str]:
    """Whole sentences packed into chunks of at most max_chars (a longer sentence is cut)."""
    sentences = re.split(r"(?<=[.!?…\"”’])\s+", " ".join(text.split()))
    chunks: list[str] = []
    current = ""
    for sentence in sentences:
        while len(sentence) > max_chars:
            if current:
                chunks.append(current)
                current = ""
            chunks.append(sentence[:max_chars])
            sentence = sentence[max_chars:]
        if current and len(current) + 1 + len(sentence) > max_chars:
            chunks.append(current)
            current = sentence
        else:
            current = f"{current} {sentence}".strip()
    if current:
        chunks.append(current)
    return chunks or [text]


def combine(parts: list[Scores]) -> Scores:
    """The strictest reading over chunks: highest score per category, a ban if any chunk sets it,
    and each category's quote from the chunk that scored it highest."""
    data: dict[str, Any] = {c: max(getattr(p, c) for p in parts) for c in CATEGORIES}
    data["real_person"] = any(p.real_person for p in parts)
    data["operational_harm"] = any(p.operational_harm for p in parts)
    quotes = []
    for c in CATEGORIES:
        if data[c]:
            top = max(parts, key=lambda p: getattr(p, c))
            quotes += [q for q in top.quotes if q.category == c][:1]
    data["quotes"] = quotes
    return Scores.model_validate(data)


class Censor:
    def __init__(self, classify: Classify, profile: str = "rubric") -> None:
        self.classify = classify
        self.profile = profile
        if profile == "rubric":
            self.rubric = (HERE / "rubric.md").read_text(encoding="utf-8")
            self.chunk_chars: int | None = None
        elif profile == "short":
            self.rubric = SHORT_SYSTEM
            self.chunk_chars = SHORT_CHUNK_CHARS
        else:
            raise ValueError(f"unknown Censor profile {profile!r}")

    def classify_text(self, text: str) -> tuple[Scores, int, int]:
        """Scores plus input and output tokens; chunked and combined for the short profile."""
        if self.chunk_chars is None or len(text) <= self.chunk_chars:
            return self._classify_one(text)
        parts, tokens_in, tokens_out = [], 0, 0
        for chunk in split_chunks(text, self.chunk_chars):
            scores, t_in, t_out = self._classify_one(chunk)
            parts.append(scores)
            tokens_in, tokens_out = tokens_in + t_in, tokens_out + t_out
        return combine(parts), tokens_in, tokens_out

    def _classify_one(self, text: str) -> tuple[Scores, int, int]:
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


def make_censor(model: str) -> Censor:
    """The Censor for a model ID, with the profile it was trained for: an MLX LoRA adapter folder
    may hold censor_profile.txt (written by training); anything else uses the rubric profile."""
    import os

    from services.llm import MLX_ADAPTERS_DIR, parse_call

    profile = "rubric"
    if model.startswith("mlx:") and "+" in model:
        adapters = Path(os.environ.get("MLX_ADAPTERS_DIR", MLX_ADAPTERS_DIR))
        path = adapters / model.split("+", 1)[1] / PROFILE_FILE
        if path.exists():
            profile = path.read_text(encoding="utf-8").strip()
    return Censor(parse_call(model, Scores, name="censor"), profile)


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
