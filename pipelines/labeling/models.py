"""Silver label records: one SceneExtraction per scene, with where it came from."""

from pydantic import BaseModel

from services.extractor.schema import SceneExtraction

SCHEMA_VERSION = 2


class SilverLabel(BaseModel):
    scene_id: str
    source: str  # who produced it: "manual-pilot", or the API model ID
    schema_version: int = SCHEMA_VERSION
    extraction: SceneExtraction
