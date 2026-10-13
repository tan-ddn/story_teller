"""Archivist output: the Story Bible facts extracted from one scene.

Every fact carries an `evidence` quote copied verbatim from the scene, so it can be checked
against the text (see `check_evidence`) and shown to the author as the source sentence.
Entity IDs are local to the scene ("e1", "e2", ...). An entity already in the Story Bible
is linked through `known_id`, taken from the known-entities context given to the model.
"""

import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

LocalId = Field(pattern=r"^e\d+$")
SnakeCase = Field(pattern=r"^[a-z][a-z0-9_]*$")

# How a scene, or one fact inside it, relates to the main story. "told" is a story, letter or
# rumour recounted by a character: it describes the story world but is not narrated as fact.
Mode = Literal["main", "flashback", "frame", "dream", "prophecy", "told"]

# Fixed vocabulary, read src -> dst. Symmetric types are listed once per pair.
RelationType = Literal[
    # family
    "parent_of",
    "grandparent_of",
    "guardian_of",
    "aunt_or_uncle_of",
    "sibling_of",  # symmetric
    "spouse_of",  # symmetric
    "engaged_to",  # symmetric
    "former_partner_of",  # symmetric
    # social
    "friend_of",  # symmetric
    "enemy_of",  # symmetric
    "rival_of",  # symmetric
    "loves",
    "works_for",
    "leader_of",
    "member_of",
    "teacher_of",
    # things and places
    "owns",
    "made",
    "lives_in",
    "located_in",
    "born_in",
    "near",  # symmetric
]

# Attribute keys with a fixed meaning. Other snake_case keys are allowed.
CORE_KEYS = {
    "life_status": "alive | dead | presumed_dead",
    "legal_status": "arrested, imprisoned, outlaw, convict, ...",
    "condition": "health, injury, or a lasting situation: wounded, lost, ruined, enchanted",
    "occupation": "job or role",
    "age": "as written",
    "appearance": "lasting looks: hair, eyes, build, clothing style",
    "trait": "character trait stated outright",
    "species": "for non-human characters",
}
LIFE_STATUS = {"alive", "dead", "presumed_dead"}


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Entity(_Strict):
    id: str = LocalId
    type: Literal["character", "location", "object", "vehicle", "org", "group"]
    name: str = Field(min_length=1)  # the fullest name used in this scene
    aliases: list[str] = []  # other names used for it in this scene: "the Ashborn", "Kael"
    known_id: str | None = None  # Story Bible ID if this is an entity from the context
    evidence: str = Field(min_length=1)


class AttributeClaim(_Strict):
    entity: str = LocalId
    key: str = SnakeCase  # a CORE_KEYS key where one fits
    value: str = Field(min_length=1)
    hedged: bool = False  # inferred, suspected, alleged or reported, not stated as fact
    mode: Mode | None = None  # set only when it differs from the scene's mode
    evidence: str = Field(min_length=1)


class Event(_Strict):
    description: str = Field(min_length=1)  # one short sentence
    participants: list[str] = Field(min_length=1)  # local entity IDs
    location: str | None = None  # local entity ID of a location or vehicle
    hedged: bool = False  # suspected, rumoured or alleged, not narrated as having happened
    mode: Mode | None = None  # set only when it differs from the scene's mode
    evidence: str = Field(min_length=1)


class Relation(_Strict):
    src: str = LocalId
    dst: str = LocalId
    type: RelationType
    hedged: bool = False
    mode: Mode | None = None
    evidence: str = Field(min_length=1)


class TimeCue(_Strict):
    kind: Literal["absolute", "relative"]  # a date or clock time, or relative to other events
    value: str = Field(min_length=1)  # as written: "the spring of 1885", "three days later"
    mode: Mode | None = None  # set only when it differs from the scene's mode
    evidence: str = Field(min_length=1)


class SceneTime(_Strict):
    mode: Mode
    cues: list[TimeCue] = []  # in the order they appear


class SceneExtraction(_Strict):
    entities: list[Entity]
    attributes: list[AttributeClaim]
    events: list[Event]
    relations: list[Relation]
    time: SceneTime

    @model_validator(mode="after")
    def _ids_resolve(self) -> "SceneExtraction":
        ids = [e.id for e in self.entities]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate entity IDs")
        types = {e.id: e.type for e in self.entities}
        refs = [a.entity for a in self.attributes]
        refs += [p for ev in self.events for p in ev.participants]
        refs += [r for rel in self.relations for r in (rel.src, rel.dst)]
        if missing := sorted(set(refs) - set(types)):
            raise ValueError(f"unknown entity IDs: {missing}")
        for ev in self.events:
            if ev.location is not None and types.get(ev.location) not in ("location", "vehicle"):
                raise ValueError(f"event location {ev.location!r} is not a location or vehicle")
        for a in self.attributes:
            if a.key == "life_status" and a.value not in LIFE_STATUS:
                raise ValueError(f"life_status must be one of {sorted(LIFE_STATUS)}")
        return self


def _squash(text: str) -> str:
    return " ".join(re.sub(r"[‘’]", "'", re.sub(r"[“”]", '"', text)).split())


def check_evidence(extraction: SceneExtraction, scene_text: str) -> list[str]:
    """Return the evidence quotes not found in the scene (whitespace and quote style ignored)."""
    scene = _squash(scene_text)
    quotes = [e.evidence for e in extraction.entities]
    quotes += [a.evidence for a in extraction.attributes]
    quotes += [ev.evidence for ev in extraction.events]
    quotes += [r.evidence for r in extraction.relations]
    quotes += [c.evidence for c in extraction.time.cues]
    return [q for q in quotes if _squash(q) not in scene]
