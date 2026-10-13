"""Silver label records: one SceneExtraction per scene, with where it came from."""

from pydantic import BaseModel

from services.extractor.schema import SceneExtraction

SCHEMA_VERSION = 2


class SilverLabel(BaseModel):
    scene_id: str
    source: str  # who produced it: "manual-pilot", or the API model ID
    schema_version: int = SCHEMA_VERSION
    extraction: SceneExtraction


class LabelRun(BaseModel):
    """One API labelling attempt for a scene; extraction is None if it failed."""

    scene_id: str
    model: str
    extraction: SceneExtraction | None = None
    error: str | None = None
    attempts: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    seconds: float = 0.0
    cost_usd: float = 0.0
