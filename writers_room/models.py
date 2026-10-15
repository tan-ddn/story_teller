"""Brainstormer output: the story layout, planned Story Bible and style sheet for one novel.

The plan is saved as JSON. The author edits it and sets `approved` to true (the human gate);
Showrunner refuses to write from a plan that is not approved.
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from services.extractor.schema import Mode

Rating = Literal["children", "teen"]


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ChapterPlan(_Strict):
    number: int = Field(ge=1)
    title: str
    pov: str  # name of a planned character
    story_time: str  # anchor as prose: "the morning after the fire", "ten years earlier"
    mode: Mode = "main"
    beats: list[str] = Field(min_length=1)  # what must happen, in order


class PlannedCharacter(_Strict):
    name: str = Field(min_length=1)
    aliases: list[str] = []
    attributes: dict[str, str] = {}  # CORE_KEYS where they fit: appearance, age, occupation
    arc: str  # how the character changes over the book


class PlannedPlace(_Strict):
    name: str = Field(min_length=1)
    description: str


class StyleSheet(_Strict):
    tense: Literal["past", "present"]
    pov_rules: str  # "close third person, one POV per chapter"
    tone: str
    voices: dict[str, str] = {}  # character name -> dialogue voice notes


class StoryPlan(_Strict):
    title: str
    premise: str
    rating: Rating
    chapters: list[ChapterPlan] = Field(min_length=1)
    characters: list[PlannedCharacter] = Field(min_length=1)
    places: list[PlannedPlace] = []
    world_rules: list[str] = []
    style: StyleSheet
    approved: bool = False  # set by the author, never by Brainstormer

    @model_validator(mode="after")
    def _consistent(self) -> "StoryPlan":
        numbers = [c.number for c in self.chapters]
        if numbers != list(range(1, len(numbers) + 1)):
            raise ValueError("chapters must be numbered 1..N in order")
        names = {c.name for c in self.characters}
        if unknown := sorted({c.pov for c in self.chapters} - names):
            raise ValueError(f"POV characters not in the plan: {unknown}")
        return self

    def character(self, name: str) -> PlannedCharacter:
        return next(c for c in self.characters if c.name == name)
