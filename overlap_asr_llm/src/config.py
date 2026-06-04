"""Configuration loading and validation."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Sample:
    id: str
    audio_path: Path
    overlap_level: str
    speakers: int
    reference: str | None = None


@dataclass(frozen=True)
class ExperimentConfig:
    project_name: str
    output_dir: Path
    language: str
    models: dict[str, str]
    rag_context: list[str]
    samples: list[Sample]
    base_dir: Path


def load_config(path: str | Path) -> ExperimentConfig:
    config_path = Path(path).resolve()
    base_dir = config_path.parent.parent
    with config_path.open("r", encoding="utf-8") as f:
        raw: dict[str, Any] = json.load(f)

    required = ["project_name", "output_dir", "language", "models", "samples"]
    missing = [key for key in required if key not in raw]
    if missing:
        raise ValueError(f"Missing required config keys: {', '.join(missing)}")

    samples = [
        Sample(
            id=item["id"],
            audio_path=(base_dir / item["audio_path"]).resolve(),
            overlap_level=item.get("overlap_level", "unknown"),
            speakers=int(item.get("speakers", 1)),
            reference=item.get("reference"),
        )
        for item in raw["samples"]
    ]

    return ExperimentConfig(
        project_name=raw["project_name"],
        output_dir=(base_dir / raw["output_dir"]).resolve(),
        language=raw["language"],
        models=dict(raw["models"]),
        rag_context=list(raw.get("rag_context", [])),
        samples=samples,
        base_dir=base_dir,
    )
