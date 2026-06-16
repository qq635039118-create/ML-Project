"""Configuration loading and validation."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ReferenceSpeaker:
    speaker: str
    text: str


@dataclass(frozen=True)
class Sample:
    id: str
    audio_path: Path
    overlap_level: str
    speakers: int
    reference: str | None = None
    reference_speakers: tuple[ReferenceSpeaker, ...] = ()
    reference_mode: str = "flat"


@dataclass(frozen=True)
class LLMRAGSource:
    label: str
    pipeline: str
    models: dict[str, str]


@dataclass(frozen=True)
class ExperimentConfig:
    project_name: str
    output_dir: Path
    language: str
    asr_prompt: str | None
    models: dict[str, str]
    asr_models: list[str]
    pipelines: list[str]
    llm_rag_sources: tuple[LLMRAGSource, ...]
    rag_context: list[str]
    samples: list[Sample]
    base_dir: Path


def load_config(path: str | Path) -> ExperimentConfig:
    config_path = Path(path).resolve()
    base_dir = config_path.parent.parent
    raw = _load_config_json(config_path)

    required = ["project_name", "output_dir", "language", "models", "samples"]
    missing = [key for key in required if key not in raw]
    if missing:
        raise ValueError(f"Missing required config keys: {', '.join(missing)}")

    default_reference_speakers = _load_reference_speakers(
        raw.get("default_reference_speakers", [])
    )
    default_reference_mode = raw.get(
        "default_reference_mode",
        "speaker_block" if default_reference_speakers else "flat",
    )
    default_reference = _reference_text(raw.get("default_reference"))

    samples = []
    for item in raw["samples"]:
        reference_speakers = _load_reference_speakers(
            item.get("reference_speakers", default_reference_speakers)
        )
        samples.append(
            Sample(
                id=item["id"],
                audio_path=(base_dir / item["audio_path"]).resolve(),
                overlap_level=item.get("overlap_level", "unknown"),
                speakers=int(item.get("speakers", 1)),
                reference=_reference_text(item.get("reference", default_reference)),
                reference_speakers=reference_speakers,
                reference_mode=item.get(
                    "reference_mode",
                    default_reference_mode if reference_speakers else "flat",
                ),
            )
        )

    return ExperimentConfig(
        project_name=raw["project_name"],
        output_dir=(base_dir / raw["output_dir"]).resolve(),
        language=raw["language"],
        asr_prompt=raw.get("asr_prompt"),
        models=dict(raw["models"]),
        asr_models=list(raw.get("asr_models", [raw["models"].get("asr", "mock")])),
        pipelines=list(
            raw.get(
                "pipelines",
                [
                    "direct_asr",
                    "diarization_asr",
                    "separation_asr",
                    "llm_rag_refine",
                ],
            )
        ),
        llm_rag_sources=_load_llm_rag_sources(raw.get("llm_rag_sources", [])),
        rag_context=list(raw.get("rag_context", [])),
        samples=samples,
        base_dir=base_dir,
    )


def _load_config_json(
    config_path: Path,
    seen: tuple[Path, ...] = (),
) -> dict[str, Any]:
    if config_path in seen:
        cycle = " -> ".join(path.as_posix() for path in (*seen, config_path))
        raise ValueError(f"Config extends cycle detected: {cycle}")

    with config_path.open("r", encoding="utf-8") as f:
        raw: dict[str, Any] = json.load(f)

    extends = raw.pop("extends", None)
    if not extends:
        return raw

    parent_paths = extends if isinstance(extends, list) else [extends]
    merged: dict[str, Any] = {}
    for parent in parent_paths:
        parent_path = (config_path.parent / str(parent)).resolve()
        parent_raw = _load_config_json(parent_path, (*seen, config_path))
        merged = _merge_config_dicts(merged, parent_raw)
    return _merge_config_dicts(merged, raw)


def _merge_config_dicts(
    base: dict[str, Any],
    override: dict[str, Any],
) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if (
            key == "models"
            and isinstance(merged.get(key), dict)
            and isinstance(value, dict)
        ):
            merged[key] = {**merged[key], **value}
        else:
            merged[key] = value
    return merged


def _reference_text(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, list):
        text = " ".join(str(part).strip() for part in value if str(part).strip())
        return text or None
    text = str(value).strip()
    return text or None


def _load_llm_rag_sources(raw: Any) -> tuple[LLMRAGSource, ...]:
    if not raw:
        return ()
    if not isinstance(raw, list):
        raise ValueError("llm_rag_sources must be a list.")

    sources = []
    for index, item in enumerate(raw, start=1):
        if isinstance(item, str):
            sources.append(
                LLMRAGSource(
                    label=item,
                    pipeline=item,
                    models={},
                )
            )
            continue
        if not isinstance(item, dict):
            raise ValueError("llm_rag_sources entries must be strings or objects.")
        pipeline = str(item.get("pipeline", "diarization_asr"))
        models = item.get("models", {})
        if not isinstance(models, dict):
            raise ValueError("llm_rag_sources models must be an object.")
        sources.append(
            LLMRAGSource(
                label=str(item.get("label", f"source_{index}")),
                pipeline=pipeline,
                models={str(key): str(value) for key, value in models.items()},
            )
        )
    return tuple(sources)


def _load_reference_speakers(raw: Any) -> tuple[ReferenceSpeaker, ...]:
    if not raw:
        return ()
    if isinstance(raw, tuple) and all(
        isinstance(item, ReferenceSpeaker) for item in raw
    ):
        return raw
    if isinstance(raw, dict):
        return tuple(
            ReferenceSpeaker(speaker=str(speaker), text=str(text))
            for speaker, text in raw.items()
        )
    if isinstance(raw, list):
        speakers = []
        for item in raw:
            if not isinstance(item, dict):
                raise ValueError("reference_speakers entries must be objects.")
            speakers.append(
                ReferenceSpeaker(
                    speaker=str(item.get("speaker", "")),
                    text=str(item.get("text", "")),
                )
            )
        return tuple(speakers)
    raise ValueError("reference_speakers must be a list or object.")
