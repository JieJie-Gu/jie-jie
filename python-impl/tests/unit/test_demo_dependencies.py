from __future__ import annotations

import tomllib
from pathlib import Path


def test_demo_optional_dependencies_include_gradio_and_requests() -> None:
    pyproject = Path(__file__).parents[2] / "pyproject.toml"
    data = tomllib.loads(pyproject.read_text(encoding="utf-8"))

    optional = data["project"]["optional-dependencies"]
    demo_dependencies = optional["demo"]

    assert "gradio>=4,<6" in demo_dependencies
    assert "requests>=2.32,<3" in demo_dependencies
