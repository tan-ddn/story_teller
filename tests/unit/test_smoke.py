import importlib

import pytest

PACKAGES = [
    "pipelines.ingest",
    "training",
    "evaluation",
    "services.api",
    "services.extractor",
    "services.checker",
    "services.mcp",
    "store",
    "writers_room",
    "guardrails",
    "security",
]


@pytest.mark.parametrize("name", PACKAGES)
def test_package_imports(name: str) -> None:
    importlib.import_module(name)
