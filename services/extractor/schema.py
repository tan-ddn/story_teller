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


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Entity(_Strict):
    id: str = LocalId
    type: Literal["character", "location", "object", "org"]
    name: str = Field(min_length=1)  # the fullest name used in this scene
    aliases: list[str] = []  # other names used for it in this scene: "the Ashborn", "Kael"
    known_id: str | None = None  # Story Bible ID if this is an entity from the context
    evidence: str = Field(min_length=1)


class AttributeClaim(_Strict):
    entity: str = LocalId
    key: str = SnakeCase  # eye_color, age, occupation, status (alive/dead), ...
    value: str = Field(min_length=1)
    evidence: str = Field(min_length=1)


class Event(_Strict):
    description: str = Field(min_length=1)  # one short sentence
    participants: list[str] = Field(min_length=1)  # local entity IDs
    location: str | None = None  # local entity ID of a location
    evidence: str = Field(min_length=1)


class Relation(_Strict):
    src: str = LocalId
    dst: str = LocalId
    type: str = SnakeCase  # sibling_of, works_for, located_in, owns, ...
    evidence: str = Field(min_length=1)


class SceneTime(_Strict):
    mode: Literal["main", "flashback", "frame", "dream", "prophecy"]
    absolute: str | None = None  # an explicit date or time, as written: "the spring of 1885"
    relative: str | None = None  # relative to another scene or event: "three days later"
    evidence: str | None = None  # None when the scene gives no time cue


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
            if ev.location is not None and types.get(ev.location) != "location":
                raise ValueError(f"event location {ev.location!r} is not a location entity")
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
    if extraction.time.evidence:
        quotes.append(extraction.time.evidence)
    return [q for q in quotes if _squash(q) not in scene]
