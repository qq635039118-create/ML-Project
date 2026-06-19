"""Launch a small speaker transcript frontend.

The app uploads one audio file, estimates overlap when needed, chooses a
pipeline from the project findings, and returns subtitle-style and grouped
speaker transcripts.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import importlib.util
import inspect
import json
import os
from pathlib import Path
import shutil
import sys
import time
from typing import Any
from urllib.parse import quote

try:
    from fastapi import FastAPI, File, Form, HTTPException, UploadFile
    from fastapi.responses import FileResponse, HTMLResponse
except ImportError:  # pragma: no cover - exercised in minimal local test envs.
    FastAPI = File = Form = HTTPException = UploadFile = None  # type: ignore[assignment]
    FileResponse = HTMLResponse = None  # type: ignore[assignment]


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
DEFAULT_ENV_FILE = ROOT / ".env"
DEFAULT_CACHE_DIR = ROOT / "outputs" / "caches"
sys.path.insert(0, str(SRC))

from overlap_asr_llm.cli import _load_env_file
from overlap_asr_llm.config import ExperimentConfig, Sample, load_config
from overlap_asr_llm.pipelines import (
    PipelineResult,
    ProviderCache,
    run_diarization_asr,
    run_diarization_turn_asr,
    run_direct_asr,
    run_separation_asr,
)
from overlap_asr_llm.providers import make_diarizer, make_llm_refiner
from overlap_asr_llm.rag import retrieve_rag_context, tags_for_sample


PIPELINE_LABELS = {
    "direct_asr": "Direct ASR",
    "diarization_asr": "Diarization + ASR alignment",
    "diarization_turn_asr": "Diarization turns + ASR",
    "separation_asr": "Speech separation + ASR",
}

OVERLAP_LEVEL_LABELS = {
    "auto": "模型自动判断",
    "none": "none overlap",
    "light": "slight overlap",
    "medium": "mild overlap",
    "heavy": "moderate overlap",
    "severe": "high overlap",
    "opposite": "severe overlap ",
}
OVERLAP_CHOICES = ["auto", "none", "light", "medium", "heavy", "severe"]
PIPELINE_CHOICES = [
    "auto",
    "direct_asr",
    "diarization_asr",
    "diarization_turn_asr",
    "separation_asr",
]
ASR_MODEL_CHOICES = [
    "faster-whisper:large-v3",
    "whisper:large-v3",
    "funasr",
    "mock",
]
DIARIZATION_MODEL_CHOICES = [
    "pyannote:pyannote/speaker-diarization-community-1",
    "speechbrain:speechbrain/spkrec-ecapa-voxceleb",
    "mock",
]
SEPARATION_MODEL_CHOICES = [
    "clearvoice:MossFormer2_SS_16K",
    "sepformer:speechbrain/sepformer-whamr16k",
    "mock",
]
LLM_MODEL_CHOICES = [
    "api",
    "api:gpt-4o-mini",
    "api:gpt-4.1-mini",
    "mock",
]
RAG_DOMAIN_CHOICES = [
    ("产品争论", "domain:product_debate"),
    ("教育", "domain:education"),
    ("医疗", "domain:healthcare"),
    ("金融", "domain:finance"),
    ("法律", "domain:legal"),
    ("技术", "domain:technology"),
    ("会议", "domain:meeting"),
    ("客服", "domain:customer_service"),
    ("研究", "domain:research"),
]
POSITIVUS_CSS = """
:root {
  --positivus-black: #191A23;
  --positivus-green: #B9FF66;
  --positivus-paper: #F3F3F3;
  --positivus-line: #191A23;
}

.gradio-container {
  background:
    radial-gradient(circle at 0% 18%, rgba(185, 255, 102, 0.62), transparent 24%),
    radial-gradient(circle at 100% 34%, rgba(185, 255, 102, 0.72), transparent 28%),
    linear-gradient(90deg, #171920 0%, #F7F7F2 18%, #F7F7F2 82%, #171920 100%) !important;
  color: var(--positivus-black) !important;
  font-family: "Space Grotesk", Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif !important;
}

.app-shell {
  max-width: 1220px;
  margin: 42px auto;
  padding: 44px;
  border: 2px solid var(--positivus-line);
  border-radius: 0;
  background: #fff;
  box-shadow: 0 18px 0 var(--positivus-black);
}

.app-shell h1 {
  display: inline-block;
  margin-bottom: 28px;
  padding: 4px 9px;
  border-radius: 7px;
  background: var(--positivus-green);
  color: var(--positivus-black);
  font-size: 36px !important;
  line-height: 1.05 !important;
  letter-spacing: 0 !important;
}

.app-shell .gr-button-primary {
  min-height: 52px;
  border: 2px solid var(--positivus-black) !important;
  border-radius: 8px !important;
  background: var(--positivus-black) !important;
  color: #fff !important;
  box-shadow: none !important;
  font-weight: 700 !important;
}

.app-shell .gr-button-primary:hover {
  background: var(--positivus-green) !important;
  color: var(--positivus-black) !important;
}

.app-shell .gr-accordion,
.app-shell .block,
.app-shell .form,
.app-shell .tabs {
  border-color: var(--positivus-black) !important;
  border-radius: 12px !important;
}

.app-shell .gr-accordion {
  border: 2px solid var(--positivus-black) !important;
  background: var(--positivus-paper) !important;
  box-shadow: 0 5px 0 var(--positivus-black);
}

.app-shell label,
.app-shell .label-wrap span {
  color: var(--positivus-black) !important;
  font-weight: 700 !important;
}

.app-shell input,
.app-shell textarea,
.app-shell select {
  border-color: var(--positivus-black) !important;
  border-radius: 8px !important;
}

.app-shell textarea {
  font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, "Liberation Mono", monospace !important;
}

.app-shell .tabitem textarea {
  background: #FFFEFA !important;
  color: #191A23 !important;
  line-height: 1.62 !important;
}

.app-shell .tab-nav button {
  border-radius: 8px 8px 0 0 !important;
  font-weight: 700 !important;
}

.app-shell .tab-nav button.selected {
  background: var(--positivus-green) !important;
  color: var(--positivus-black) !important;
}

@media (max-width: 768px) {
  .app-shell {
    margin: 16px;
    padding: 20px;
    box-shadow: 0 9px 0 var(--positivus-black);
  }

  .app-shell h1 {
    font-size: 28px !important;
  }
}
"""


def _module_available(module_name: str) -> bool:
    try:
        return importlib.util.find_spec(module_name) is not None
    except ModuleNotFoundError:
        return False


def _available_asr_choices() -> list[str]:
    choices = []
    if _module_available("faster_whisper"):
        choices.append("faster-whisper:large-v3")
    if _module_available("whisper"):
        choices.append("whisper:large-v3")
    if _module_available("funasr"):
        choices.append("funasr")
    choices.append("mock")
    return choices


def _available_diarization_choices() -> list[str]:
    choices = []
    if _module_available("pyannote.audio"):
        choices.append("pyannote:pyannote/speaker-diarization-community-1")
    if _module_available("speechbrain"):
        choices.append("speechbrain:speechbrain/spkrec-ecapa-voxceleb")
    choices.append("mock")
    return choices


def _available_separation_choices() -> list[str]:
    choices = []
    if _module_available("clearvoice"):
        choices.append("clearvoice:MossFormer2_SS_16K")
    if _module_available("speechbrain"):
        choices.append("sepformer:speechbrain/sepformer-whamr16k")
    choices.append("mock")
    return choices


def _available_llm_choices() -> list[str]:
    choices = []
    if _module_available("httpx"):
        choices.append("api")
        env_model = os.environ.get("OPENAI_MODEL", "").strip()
        if env_model and env_model not in {"api", "mock"}:
            choices.append(f"api:{env_model}")
    choices.append("mock")
    return choices


def _default_choice(choices: list[str], preferred: str) -> str:
    return preferred if preferred in choices else choices[0]


def _ui_model_choices(choices: list[str]) -> list[str]:
    visible = [choice for choice in choices if choice != "mock"]
    return visible or choices


def _launch_app(app, *, host: str, port: int, share: bool):
    launch_kwargs = {
        "server_name": host,
        "server_port": port,
        "share": share,
    }
    if "css" in inspect.signature(app.launch).parameters:
        launch_kwargs["css"] = POSITIVUS_CSS
    return app.launch(**launch_kwargs)


def _format_timestamp(seconds: float, srt: bool = False) -> str:
    seconds = max(seconds, 0.0)
    minutes, second_value = divmod(seconds, 60)
    hours, minute_value = divmod(int(minutes), 60)
    if srt:
        whole_seconds = int(second_value)
        milliseconds = int(round((second_value - whole_seconds) * 1000))
        if milliseconds == 1000:
            whole_seconds += 1
            milliseconds = 0
        return f"{hours:02d}:{minute_value:02d}:{whole_seconds:02d},{milliseconds:03d}"
    return f"{hours:02d}:{minute_value:02d}:{second_value:06.3f}"


def _ordered_segments(segments: list[dict[str, object]]) -> list[dict[str, object]]:
    return sorted(
        [segment for segment in segments if str(segment.get("text", "")).strip()],
        key=lambda item: (
            float(item.get("start", 0.0)),
            float(item.get("end", 0.0)),
            str(item.get("speaker", "")),
        ),
    )


def subtitle_text(segments: list[dict[str, object]]) -> str:
    blocks = []
    for index, segment in enumerate(_ordered_segments(segments), start=1):
        start = _format_timestamp(float(segment.get("start", 0.0)), srt=True)
        end = _format_timestamp(float(segment.get("end", 0.0)), srt=True)
        speaker = str(segment.get("speaker", "UNKNOWN"))
        text = str(segment.get("text", "")).strip()
        blocks.append(f"{index}\n{start} --> {end}\n[{speaker}] {text}")
    return "\n\n".join(blocks)


def speaker_joined_text(segments: list[dict[str, object]]) -> str:
    by_speaker: dict[str, list[str]] = {}
    for segment in _ordered_segments(segments):
        speaker = str(segment.get("speaker", "UNKNOWN"))
        text = str(segment.get("text", "")).strip()
        by_speaker.setdefault(speaker, []).append(text)
    return "\n\n".join(
        f"{speaker}:\n{' '.join(parts).strip()}"
        for speaker, parts in by_speaker.items()
        if " ".join(parts).strip()
    )


def _write_output_files(output_dir: Path, subtitle: str, speaker_text: str) -> tuple[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    subtitle_path = output_dir / "speaker_transcript.srt"
    speaker_path = output_dir / "speaker_blocks.txt"
    subtitle_path.write_text(subtitle, encoding="utf-8")
    speaker_path.write_text(speaker_text, encoding="utf-8")
    return str(subtitle_path), str(speaker_path)


def _download_item(path: str | Path, label: str) -> dict[str, str]:
    resolved = Path(path).resolve()
    return {
        "label": label,
        "path": str(resolved),
        "url": f"/api/download?path={quote(resolved.as_posix())}",
    }


def _collect_separated_audio_files(result: PipelineResult) -> list[dict[str, str]]:
    files: list[dict[str, str]] = []
    seen: set[str] = set()
    for segment in result.segments:
        raw_path = segment.get("separated_audio_path")
        if not raw_path:
            continue
        path = Path(str(raw_path)).resolve()
        if not path.exists() or str(path) in seen:
            continue
        seen.add(str(path))
        speaker = str(segment.get("speaker", f"speaker_{len(files) + 1}"))
        files.append(_download_item(path, speaker))
    return files


def _cleanup_previous_speaker_app_runs(
    current_output_dir: Path,
    current_upload_dir: Path | None = None,
    speaker_app_dir: Path | None = None,
) -> list[str]:
    app_dir = (speaker_app_dir or ROOT / "outputs" / "speaker_app").resolve()
    if not app_dir.exists():
        return []

    current_output = current_output_dir.resolve()
    current_upload = current_upload_dir.resolve() if current_upload_dir else None
    deleted: list[str] = []

    def remove_path(path: Path) -> None:
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()
        deleted.append(str(path))

    for item in app_dir.iterdir():
        resolved = item.resolve()
        if resolved == current_output or current_output.is_relative_to(resolved):
            continue
        if item.name == "uploads" and item.is_dir():
            for upload_item in item.iterdir():
                upload_resolved = upload_item.resolve()
                if current_upload and (
                    upload_resolved == current_upload
                    or current_upload.is_relative_to(upload_resolved)
                ):
                    continue
                remove_path(upload_item)
            continue
        remove_path(item)
    return deleted


def _timeline_text(segments: list[dict[str, object]]) -> str:
    return " ".join(
        str(segment.get("text", "")).strip() for segment in _ordered_segments(segments)
    ).strip()


def _configured_env_file() -> Path:
    return Path(os.environ.get("OVERLAP_ASR_LLM_ENV_FILE", DEFAULT_ENV_FILE)).expanduser()


def _configure_default_cache() -> Path:
    cache_dir = Path(
        os.environ.get("OVERLAP_ASR_LLM_CACHE_DIR", DEFAULT_CACHE_DIR)
    ).expanduser()
    if not cache_dir.is_absolute():
        cache_dir = (ROOT / cache_dir).resolve()
    else:
        cache_dir = cache_dir.resolve()
    os.environ["OVERLAP_ASR_LLM_CACHE_DIR"] = str(cache_dir)
    os.environ["HF_HOME"] = str(cache_dir / "huggingface")
    os.environ["HF_HUB_CACHE"] = str(cache_dir / "huggingface" / "hub")
    os.environ.setdefault("MODELSCOPE_CACHE", str(cache_dir / "modelscope"))
    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir / "matplotlib"))
    return cache_dir


def _load_frontend_env(env_file: str | Path | None = None) -> Path:
    env_path = Path(env_file).expanduser() if env_file else _configured_env_file()
    os.environ["OVERLAP_ASR_LLM_ENV_FILE"] = str(env_path)
    _load_env_file(env_path)
    _configure_default_cache()
    return env_path


def _env_loaded_summary(env_path: Path) -> str:
    present_keys = [
        key
        for key in (
            "HF_TOKEN",
            "HUGGINGFACE_TOKEN",
            "OPENAI_API_KEY",
            "OPENAI_BASE_URL",
            "OPENAI_MODEL",
        )
        if os.environ.get(key)
    ]
    if not env_path.exists():
        return f"Env file not found: {env_path}"
    if present_keys:
        return f"Loaded env file: {env_path} ({', '.join(present_keys)} set)"
    return f"Loaded env file: {env_path} (no known token keys set)"


def _env_diagnostics() -> dict[str, object]:
    cache_root = _configure_default_cache()
    pyannote_cache = (
        cache_root
        / "huggingface"
        / "models--pyannote--speaker-diarization-community-1"
    )
    return {
        "env_file": str(_configured_env_file()),
        "hf_token_set": bool(os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")),
        "hf_endpoint": os.environ.get("HF_ENDPOINT", ""),
        "hf_home": os.environ.get("HF_HOME", ""),
        "hf_hub_cache": os.environ.get("HF_HUB_CACHE", ""),
        "pyannote_cache_exists": pyannote_cache.exists(),
        "pyannote_cache": str(pyannote_cache),
    }


def _load_base_config(config_path: Path, mock: bool) -> ExperimentConfig:
    _load_frontend_env()
    config = load_config(config_path)
    if mock:
        config.models.update(
            {"asr": "mock", "diarization": "mock", "separation": "mock", "llm": "mock"}
        )
    return config


def _runtime_config(
    base_config: ExperimentConfig,
    audio_path: Path,
    output_dir: Path,
    speakers: int,
    language: str,
    asr_model: str,
    diarization_model: str,
    separation_model: str,
    llm_model: str,
    prompt: str | None,
    overlap_level: str,
) -> tuple[ExperimentConfig, Sample]:
    sample = Sample(
        id=f"upload_{audio_path.stem}",
        audio_path=audio_path.resolve(),
        overlap_level=overlap_level,
        speakers=max(1, int(speakers)),
    )
    models = {
        **base_config.models,
        "asr": asr_model.strip() or base_config.models.get("asr", "mock"),
        "diarization": diarization_model.strip()
        or base_config.models.get("diarization", "mock"),
        "separation": separation_model.strip()
        or base_config.models.get("separation", "mock"),
        "llm": llm_model.strip() or base_config.models.get("llm", "mock"),
    }
    config = replace(
        base_config,
        output_dir=output_dir,
        language=language.strip() or base_config.language,
        asr_prompt=prompt,
        models=models,
        pipelines=[],
        samples=[sample],
    )
    return config, sample


def _turn_overlap_ratio(turns: list[dict[str, object]]) -> float:
    events = []
    for turn in turns:
        start = float(turn.get("start", 0.0))
        end = float(turn.get("end", start))
        if end <= start:
            continue
        events.append((start, 1))
        events.append((end, -1))
    if not events:
        return 0.0

    events.sort(key=lambda item: (item[0], -item[1]))
    active = 0
    previous = events[0][0]
    union_duration = 0.0
    overlap_duration = 0.0
    for timestamp, delta in events:
        duration = max(0.0, timestamp - previous)
        if active > 0:
            union_duration += duration
        if active > 1:
            overlap_duration += duration
        active += delta
        previous = timestamp
    if union_duration <= 0:
        return 0.0
    return overlap_duration / union_duration


def _level_from_ratio(ratio: float) -> str:
    if ratio >= 0.35:
        return "severe"
    if ratio >= 0.12:
        return "heavy"
    if ratio >= 0.05:
        return "medium"
    if ratio >= 0.015:
        return "light"
    return "none"


def _speaker_count_from_turns(turns: list[dict[str, object]]) -> int:
    labels = {
        str(turn.get("speaker", "")).strip()
        for turn in turns
        if str(turn.get("speaker", "")).strip()
        and str(turn.get("speaker", "")).strip() != "UNKNOWN"
    }
    return max(1, len(labels))


def _estimate_audio_profile(
    audio_path: Path,
    diarization_model: str,
    fallback_speakers: int = 2,
) -> tuple[str, float | None, int, str]:
    try:
        diarizer = make_diarizer(diarization_model)
        try:
            turns = diarizer.diarize(audio_path, 0)
        except TypeError:
            turns = diarizer.diarize(audio_path, max(1, int(fallback_speakers)))
        if hasattr(diarizer, "release_gpu"):
            diarizer.release_gpu()
        ratio = _turn_overlap_ratio(turns)
        speaker_count = _speaker_count_from_turns(turns)
        if speaker_count <= 1 and fallback_speakers > 1 and not turns:
            speaker_count = max(1, int(fallback_speakers))
        return _level_from_ratio(ratio), ratio, speaker_count, ""
    except Exception as exc:
        return "medium", None, max(1, int(fallback_speakers)), f"{type(exc).__name__}: {exc}"


def _estimate_overlap_level(
    audio_path: Path,
    diarization_model: str,
    speakers: int,
) -> tuple[str, float | None, str]:
    level, ratio, _speaker_count, error = _estimate_audio_profile(
        audio_path=audio_path,
        diarization_model=diarization_model,
        fallback_speakers=speakers,
    )
    return level, ratio, error


def _auto_overlap_error_summary(
    diarization_model: str,
    error: str,
) -> str:
    payload = {
        "selected_pipeline": None,
        "selected_label": None,
        "overlap_level": None,
        "overlap_ratio": None,
        "speaker_count": None,
        "overlap_estimation_error": error,
        "selection_reason": (
            "auto OVR requires a working diarization model before a pipeline can be selected"
        ),
        "ran_pipeline_count": 0,
        "diarization_model": diarization_model,
        "env_diagnostics": _env_diagnostics(),
        "next_steps": [
            "Connect to Hugging Face once so the diarization model can be downloaded and cached.",
            "Set HF_TOKEN if using pyannote and accept the model terms on Hugging Face.",
            "Or manually choose Overlap/Pipeline if you already know the audio condition.",
            "Or switch Diarization to a model that is already cached locally.",
        ],
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def recommended_pipeline(overlap_level: str, speakers: int) -> str:
    if speakers <= 1:
        return "direct_asr"
    if overlap_level == "none":
        return "diarization_turn_asr"
    if overlap_level in {"severe", "opposite"}:
        return "separation_asr"
    if overlap_level in {"light", "medium", "heavy"}:
        return "diarization_asr"
    return "diarization_asr"


def recommended_pipelines(overlap_level: str, speakers: int) -> list[str]:
    return [recommended_pipeline(overlap_level, speakers)]


def _selection_reason(overlap_level: str, speakers: int, pipeline: str) -> str:
    if speakers <= 1:
        return "single-speaker audio uses direct ASR"
    if overlap_level == "none":
        return "project results showed diarization_turn_asr is strongest for no-overlap audio"
    if overlap_level in {"severe", "opposite"}:
        return "project results showed separation_asr is strongest for extremely overlapped audio"
    if overlap_level in {"light", "medium", "heavy"}:
        return "project results showed diarization_asr has the best average CER/WER and readability"
    return f"defaulting to {pipeline} for unknown overlap level"


def _run_pipeline(
    pipeline: str,
    config: ExperimentConfig,
    sample: Sample,
    providers: ProviderCache,
) -> PipelineResult:
    if pipeline == "direct_asr":
        return run_direct_asr(config, sample, providers)
    if pipeline == "diarization_asr":
        return run_diarization_asr(config, sample, providers)
    if pipeline == "diarization_turn_asr":
        return run_diarization_turn_asr(config, sample, providers)
    if pipeline == "separation_asr":
        return run_separation_asr(config, sample, providers)
    raise ValueError(f"Unsupported pipeline: {pipeline}")


def _result_quality_key(result: PipelineResult) -> tuple[int, int, float]:
    if result.error:
        return (0, 0, 0.0)
    speaker_count = len([label for label in result.speaker_labels.split(",") if label])
    text_length = len(_timeline_text(result.segments) or result.text)
    return (1, speaker_count, float(text_length))


def _select_result(results: list[PipelineResult], preferred_order: list[str]) -> PipelineResult:
    successful = [result for result in results if not result.error]
    if not successful:
        return results[0]
    order = {name: index for index, name in enumerate(preferred_order)}
    return max(
        successful,
        key=lambda result: (
            _result_quality_key(result),
            -order.get(result.pipeline, len(preferred_order)),
        ),
    )


def _pipeline_summary(
    selected: PipelineResult,
    results: list[PipelineResult],
    overlap_level: str,
    overlap_ratio: float | None,
    speaker_count: int,
    overlap_error: str,
    selection_reason: str,
) -> str:
    rows: list[dict[str, Any]] = []
    for result in results:
        rows.append(
            {
                "pipeline": result.pipeline,
                "label": PIPELINE_LABELS.get(result.pipeline, result.pipeline),
                "selected": result is selected,
                "model": result.model,
                "speaker_labels": result.speaker_labels,
                "runtime_seconds": result.runtime_seconds,
                "segments_with_text": len(_ordered_segments(result.segments)),
                "error": result.error,
            }
        )
    payload = {
        "selected_pipeline": selected.pipeline,
        "selected_label": PIPELINE_LABELS.get(selected.pipeline, selected.pipeline),
        "overlap_level": overlap_level,
        "overlap_label": OVERLAP_LEVEL_LABELS.get(overlap_level, overlap_level),
        "overlap_ratio": None if overlap_ratio is None else round(overlap_ratio, 4),
        "speaker_count": speaker_count,
        "overlap_estimation_error": overlap_error,
        "selection_reason": selection_reason,
        "ran_pipeline_count": len(results),
        "candidates": rows,
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def _refine_with_llm(
    subtitle: str,
    overlap_level: str,
    llm_model: str,
    rag_domains: list[str] | None = None,
    llm_extra_prompt: str | None = None,
) -> str:
    tags = tags_for_sample(overlap_level, "speaker_transcript")
    tags.extend(rag_domains or [])
    base_context = []
    if llm_extra_prompt and llm_extra_prompt.strip():
        base_context.append(f"User extra LLM guidance: {llm_extra_prompt.strip()}")
    context = retrieve_rag_context(tags, base_context=base_context)
    source_text = (
        "Retrieved context:\n"
        + "\n".join(f"- {item}" for item in context)
        + "\n\nSpeaker transcript:\n"
        + subtitle
    )
    refiner_kind = llm_model.strip() or "api"
    return make_llm_refiner(refiner_kind).refine(source_text, context)


def _as_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return []


def _format_llm_output(raw_text: str) -> dict[str, Any]:
    if not raw_text.strip():
        return {
            "raw": "",
            "refined_text": "",
            "changes": [],
            "uncertain_spans": [],
            "hallucination_risk": "",
            "display_text": "",
        }
    try:
        data = json.loads(raw_text)
    except json.JSONDecodeError:
        display = raw_text.strip()
        return {
            "raw": raw_text,
            "refined_text": display,
            "changes": [],
            "uncertain_spans": [],
            "hallucination_risk": "unknown",
            "display_text": display,
        }
    if not isinstance(data, dict):
        display = str(data)
        return {
            "raw": raw_text,
            "refined_text": display,
            "changes": [],
            "uncertain_spans": [],
            "hallucination_risk": "unknown",
            "display_text": display,
        }

    refined = str(data.get("refined_text", "")).strip()
    changes = _as_list(data.get("changes"))
    uncertain = _as_list(data.get("uncertain_spans"))
    risk = str(data.get("hallucination_risk", "unknown")).strip() or "unknown"
    lines = [f"幻觉风险: {risk}"]
    if changes:
        lines.append("\n关键优化:")
        lines.extend(f"- {item}" for item in changes[:6])
    if uncertain:
        lines.append("\n需要人工留意:")
        lines.extend(f"- {item}" for item in uncertain[:6])
    if refined:
        lines.append("\n调优后文本:\n" + refined)
    return {
        "raw": raw_text,
        "refined_text": refined,
        "changes": changes,
        "uncertain_spans": uncertain,
        "hallucination_risk": risk,
        "display_text": "\n".join(lines).strip(),
    }


def run_transcription(
    audio_path: str | Path,
    overlap_level_choice: str,
    pipeline_choice: str,
    speakers: int,
    language: str,
    asr_model: str,
    diarization_model: str,
    separation_model: str,
    hotwords: str,
    rag_domains: list[str],
    llm_extra_prompt: str,
    use_llm: bool,
    llm_model: str,
    mock: bool,
    current_upload_dir: Path | None = None,
) -> dict[str, Any]:
    audio_path = Path(audio_path)
    prompt = hotwords.strip() or None
    base_config = _load_base_config(ROOT / "configs" / "all_pipelines.json", mock)
    if mock:
        asr_model = "mock"
        diarization_model = "mock"
        separation_model = "mock"
        llm_model = "mock"
    run_id = time.strftime("%Y%m%d_%H%M%S")
    output_dir = ROOT / "outputs" / "speaker_app" / f"{run_id}_{audio_path.stem}"

    overlap_error = ""
    overlap_ratio = None
    inferred_speakers = max(1, int(speakers))
    selected_overlap_level = overlap_level_choice
    ovr_model = diarization_model.strip() or base_config.models.get("diarization", "mock")
    profiled_overlap_level, overlap_ratio, inferred_speakers, overlap_error = (
        _estimate_audio_profile(
            audio_path=audio_path,
            diarization_model=ovr_model,
            fallback_speakers=max(1, int(speakers)),
        )
    )
    if selected_overlap_level == "auto":
        selected_overlap_level = profiled_overlap_level
    if overlap_error and overlap_level_choice == "auto":
        summary = _auto_overlap_error_summary(ovr_model, overlap_error)
        message = (
            "模型未能完成 OVR / 说话人数推断，所以没有运行 pipeline。\n\n"
            f"Diarization model: {ovr_model}\n"
            f"Error: {overlap_error}"
        )
        return {
            "ok": False,
            "summary": json.loads(summary),
            "summary_text": summary,
            "subtitle": message,
            "speaker_text": message,
            "llm_text": "",
            "llm": _format_llm_output(""),
            "downloads": [],
            "separated_audio": [],
        }

    config, sample = _runtime_config(
        base_config=base_config,
        audio_path=audio_path,
        output_dir=output_dir,
        speakers=inferred_speakers,
        language=language,
        asr_model=asr_model,
        diarization_model=diarization_model,
        separation_model=separation_model,
        llm_model=llm_model,
        prompt=prompt,
        overlap_level=selected_overlap_level,
    )

    selected_pipeline = (
        recommended_pipeline(selected_overlap_level, inferred_speakers)
        if pipeline_choice == "auto"
        else pipeline_choice
    )
    selection_reason = (
        _selection_reason(selected_overlap_level, inferred_speakers, selected_pipeline)
        if pipeline_choice == "auto"
        else "manual pipeline selection"
    )

    providers = ProviderCache(config)
    results = [_run_pipeline(selected_pipeline, config, sample, providers)]
    selected = results[0]
    subtitle = subtitle_text(selected.segments)
    if not subtitle:
        subtitle = selected.text
    speaker_text = speaker_joined_text(selected.segments)
    if not speaker_text:
        speaker_text = selected.text
    subtitle_file, speaker_file = _write_output_files(output_dir, subtitle, speaker_text)

    llm_text = ""
    llm = _format_llm_output("")
    if use_llm:
        try:
            llm_text = _refine_with_llm(
                subtitle,
                selected_overlap_level,
                llm_model,
                rag_domains=rag_domains,
                llm_extra_prompt=llm_extra_prompt,
            )
        except Exception as exc:
            llm_text = f"LLM/RAG refinement error: {type(exc).__name__}: {exc}"
        llm = _format_llm_output(llm_text)

    summary = _pipeline_summary(
        selected=selected,
        results=results,
        overlap_level=selected_overlap_level,
        overlap_ratio=overlap_ratio,
        speaker_count=inferred_speakers,
        overlap_error=overlap_error,
        selection_reason=selection_reason,
    )
    summary_data = json.loads(summary)
    subtitle_item = _download_item(subtitle_file, "SRT 字幕文件")
    speaker_item = _download_item(speaker_file, "说话人汇总文本")
    separated_audio = _collect_separated_audio_files(selected)
    deleted_previous_runs = _cleanup_previous_speaker_app_runs(
        output_dir,
        current_upload_dir=current_upload_dir,
    )
    summary_data["deleted_previous_run_count"] = len(deleted_previous_runs)
    return {
        "ok": selected.error is None,
        "summary": summary_data,
        "summary_text": summary,
        "subtitle": subtitle,
        "speaker_text": speaker_text,
        "llm_text": llm["display_text"],
        "llm": llm,
        "downloads": [subtitle_item, speaker_item],
        "separated_audio": separated_audio,
        "deleted_previous_runs": deleted_previous_runs,
    }


def process_audio(
    audio_file: str,
    overlap_level_choice: str,
    pipeline_choice: str,
    speakers: int,
    language: str,
    asr_model: str,
    diarization_model: str,
    separation_model: str,
    hotwords: str,
    rag_domains: list[str],
    llm_extra_prompt: str,
    use_llm: bool,
    llm_model: str,
    mock: bool,
) -> tuple[str, str, str, str, str | None, str | None]:
    if not audio_file:
        return "请先上传音频。", "", "", "", None, None
    result = run_transcription(
        audio_file,
        overlap_level_choice,
        pipeline_choice,
        speakers,
        language,
        asr_model,
        diarization_model,
        separation_model,
        hotwords,
        rag_domains,
        llm_extra_prompt,
        use_llm,
        llm_model,
        mock,
    )
    downloads = result.get("downloads", [])
    subtitle_file = downloads[0]["path"] if len(downloads) > 0 else None
    speaker_file = downloads[1]["path"] if len(downloads) > 1 else None
    return (
        str(result.get("summary_text", "")),
        str(result.get("subtitle", "")),
        str(result.get("speaker_text", "")),
        str(result.get("llm_text", "")),
        subtitle_file,
        speaker_file,
    )


WEB_INDEX_HTML = """
<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Overlap ASR Transcript Workbench</title>
  <style>
    :root {
      --ink: #191A23;
      --green: #A7F3D0;
      --amber: #FDE68A;
      --paper: #F8F6EF;
      --muted: #62646A;
      --line: #191A23;
      --soft: #F1F5F2;
      --danger: #B42318;
      --blue: #BFDBFE;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      color: var(--ink);
      background: linear-gradient(180deg, #FCFBF7 0%, var(--paper) 58%, #EEF6F1 100%);
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    .shell {
      width: min(1240px, calc(100% - 32px));
      margin: 32px auto;
      padding: 36px;
      background: #fff;
      border: 2px solid var(--line);
      border-radius: 8px;
      box-shadow: 0 14px 0 var(--ink);
    }
    header {
      display: flex;
      justify-content: space-between;
      gap: 24px;
      align-items: flex-start;
      margin-bottom: 28px;
    }
    h1 {
      margin: 0 0 12px;
      font-size: clamp(30px, 4vw, 56px);
      line-height: 1.02;
      letter-spacing: 0;
      max-width: 720px;
    }
    .tag {
      display: inline-block;
      padding: 4px 9px;
      border-radius: 7px;
      background: var(--green);
      font-weight: 800;
    }
    .subtitle {
      margin: 0;
      max-width: 720px;
      color: var(--muted);
      font-size: 16px;
      line-height: 1.5;
    }
    .status-pill {
      min-width: 188px;
      padding: 14px 16px;
      border: 2px solid var(--ink);
      border-radius: 8px;
      background: var(--soft);
      font-weight: 800;
      text-align: center;
    }
    .grid {
      display: grid;
      grid-template-columns: minmax(300px, 390px) 1fr;
      gap: 24px;
      align-items: start;
    }
    .panel {
      border: 2px solid var(--ink);
      border-radius: 8px;
      background: var(--soft);
      box-shadow: 0 6px 0 var(--ink);
      padding: 20px;
    }
    .input-panel {
      position: sticky;
      top: 20px;
    }
    .result-panel {
      background: #fff;
    }
    .result-panel.loading .result-body {
      display: none;
    }
    .result-loading {
      display: none;
      min-height: 430px;
      align-items: center;
      justify-content: center;
      border: 2px dashed var(--ink);
      border-radius: 8px;
      background: #FFFEFA;
      text-align: center;
    }
    .result-panel.loading .result-loading {
      display: flex;
    }
    .spinner {
      width: 54px;
      height: 54px;
      margin: 0 auto 18px;
      border: 5px solid #D9E2DC;
      border-top-color: var(--ink);
      border-radius: 50%;
      animation: spin 0.9s linear infinite;
    }
    .loading-title {
      font-weight: 900;
      font-size: 18px;
    }
    .loading-subtitle {
      margin-top: 8px;
      color: var(--muted);
      font-weight: 700;
    }
    @keyframes spin {
      to { transform: rotate(360deg); }
    }
    .section-title {
      display: inline-block;
      margin: 0 0 16px;
      padding: 3px 8px;
      border-radius: 6px;
      background: var(--green);
      font-size: 20px;
      font-weight: 850;
    }
    label {
      display: block;
      margin: 14px 0 6px;
      font-weight: 800;
      font-size: 13px;
    }
    input, select, textarea {
      width: 100%;
      border: 2px solid var(--ink);
      border-radius: 8px;
      background: #fff;
      color: var(--ink);
      padding: 11px 12px;
      font: inherit;
    }
    textarea { min-height: 82px; resize: vertical; }
    .row {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 12px;
    }
    details {
      margin-top: 18px;
      border: 2px solid var(--ink);
      border-radius: 8px;
      background: #fff;
    }
    summary {
      padding: 12px;
      cursor: pointer;
      font-weight: 850;
    }
    .details-body {
      padding: 0 12px 12px;
      border-top: 2px solid var(--ink);
    }
    .checkboxes {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 8px;
      margin-top: 8px;
    }
    .check {
      display: flex;
      align-items: center;
      gap: 7px;
      padding: 8px;
      border: 1.5px solid var(--ink);
      border-radius: 6px;
      background: #fff;
      font-size: 13px;
      font-weight: 700;
    }
    .check input { width: auto; }
    button {
      width: 100%;
      margin-top: 18px;
      padding: 15px 18px;
      border: 2px solid var(--ink);
      border-radius: 8px;
      background: var(--ink);
      color: #fff;
      font-weight: 900;
      cursor: pointer;
    }
    button:hover { background: var(--green); color: var(--ink); }
    button:disabled { opacity: .55; cursor: wait; }
    .cards {
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 12px;
      margin-bottom: 18px;
    }
    .metric {
      border: 2px solid var(--ink);
      border-radius: 8px;
      padding: 12px;
      background: #fff;
      min-height: 78px;
    }
    .metric:nth-child(1) { background: var(--green); }
    .metric:nth-child(2) { background: var(--blue); }
    .metric:nth-child(3) { background: var(--amber); }
    .metric span {
      display: block;
      color: var(--muted);
      font-size: 12px;
      font-weight: 800;
    }
    .metric strong {
      display: block;
      margin-top: 6px;
      font-size: 18px;
      line-height: 1.15;
      word-break: break-word;
    }
    .tabs {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      border-bottom: 2px solid var(--ink);
      margin-bottom: 14px;
    }
    .tab {
      width: auto;
      margin: 0;
      padding: 10px 14px;
      border-radius: 9px 9px 0 0;
      background: #fff;
      color: var(--ink);
      border-bottom: 0;
    }
    .tab.active { background: var(--green); }
    .view { display: none; }
    .view.active { display: block; }
    pre {
      margin: 0;
      min-height: 430px;
      max-height: 62vh;
      overflow: auto;
      padding: 18px;
      border: 2px solid var(--ink);
      border-radius: 8px;
      background: #FFFEFA;
      color: var(--ink);
      white-space: pre-wrap;
      line-height: 1.55;
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
    }
    .llm-brief {
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 10px;
      margin-bottom: 12px;
    }
    .brief-card {
      border: 2px solid var(--ink);
      border-radius: 8px;
      background: #fff;
      padding: 12px;
      min-height: 72px;
    }
    .brief-card span {
      display: block;
      color: var(--muted);
      font-size: 12px;
      font-weight: 800;
    }
    .brief-card strong {
      display: block;
      margin-top: 6px;
      font-size: 17px;
      line-height: 1.2;
    }
    .downloads, .audio-list {
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      margin: 14px 0;
    }
    .download, .audio-card {
      border: 2px solid var(--ink);
      border-radius: 8px;
      background: #fff;
      padding: 12px;
    }
    .download {
      color: var(--ink);
      font-weight: 850;
      text-decoration: none;
    }
    .audio-card { min-width: 260px; flex: 1; }
    .audio-card strong { display: block; margin-bottom: 8px; }
    audio { width: 100%; }
    .empty {
      padding: 42px;
      border: 2px dashed var(--ink);
      border-radius: 8px;
      background: var(--soft);
      color: var(--muted);
      font-weight: 700;
      text-align: center;
    }
    .error { color: var(--danger); }
    @media (max-width: 900px) {
      .shell { padding: 20px; box-shadow: 0 9px 0 var(--ink); }
      header, .grid { display: block; }
      .status-pill { margin-top: 16px; }
      .result-panel { margin-top: 20px; }
      .cards { grid-template-columns: 1fr 1fr; }
      .llm-brief { grid-template-columns: 1fr; }
      .row, .checkboxes { grid-template-columns: 1fr; }
      .input-panel { position: static; }
    }
  </style>
</head>
<body>
  <main class="shell">
    <header>
      <div>
        <h1><span class="tag">Overlap ASR</span> transcript workbench</h1>
        <p class="subtitle">上传音频后由模型估计 OVR 和说话人数，再按实验结论选择 pipeline。结果区会给出字幕、说话人汇总、LLM 可读性优化摘要和可下载文件。</p>
      </div>
      <div class="status-pill" id="status">Ready</div>
    </header>

    <div class="grid">
      <form class="panel input-panel" id="runForm">
        <h2 class="section-title">输入</h2>
        <label>音频文件</label>
        <input name="audio" type="file" accept="audio/*,.wav,.mp3,.m4a,.flac,.ogg" required />
        <input name="overlap_level" type="hidden" value="auto" />
        <input name="pipeline" type="hidden" value="auto" />
        <input name="speakers" type="hidden" value="2" />

        <label>语言</label>
        <select name="language">
          <option value="zh">中文</option>
          <option value="en">English</option>
          <option value="ja">日本語</option>
          <option value="ko">한국어</option>
          <option value="auto">Auto</option>
        </select>

        <h2 class="section-title" style="margin-top:20px">模型</h2>
        <label>ASR</label><select name="asr_model" id="asr_model"></select>
        <label>Diarization</label><select name="diarization_model" id="diarization_model"></select>
        <label>Separation</label><select name="separation_model" id="separation_model"></select>
        <label>LLM</label><select name="llm_model" id="llm_model"></select>

        <label>ASR 热词 / Prompt</label>
        <textarea name="hotwords" placeholder="例如：产品升级、教育理念、用户体验"></textarea>
        <label class="check" style="margin-top:14px"><input name="use_llm" type="checkbox" /> 使用 LLM/RAG 清理文本</label>
        <details>
          <summary>高级 RAG 和提示词</summary>
          <div class="details-body">
            <label>RAG 知识领域</label>
            <div class="checkboxes" id="rag_domains"></div>
            <label>补充给 LLM 的提示词</label>
            <textarea name="llm_extra_prompt" placeholder="例如：保持说话人标签，不要补写没听清的词"></textarea>
          </div>
        </details>
        <button id="submitButton" type="submit">上传并开始处理</button>
      </form>

      <section class="panel result-panel" id="resultPanel">
        <h2 class="section-title">结果</h2>
        <div class="cards">
          <div class="metric"><span>Pipeline</span><strong id="mPipeline">-</strong></div>
          <div class="metric"><span>OVR</span><strong id="mOvr">-</strong></div>
          <div class="metric"><span>说话人数</span><strong id="mSpeakers">-</strong></div>
          <div class="metric"><span>Status</span><strong id="mStatus">Waiting</strong></div>
        </div>
        <div class="result-loading" id="resultLoading" aria-live="polite">
          <div>
            <div class="spinner" aria-hidden="true"></div>
            <div class="loading-title">正在处理音频</div>
            <div class="loading-subtitle">模型正在估计 OVR、说话人数并生成结果</div>
          </div>
        </div>
        <div class="result-body" id="resultBody">
          <div class="downloads" id="downloads"></div>
          <div class="audio-list" id="audioList"></div>
          <div class="tabs">
            <button class="tab active" type="button" data-view="subtitleView">字幕</button>
            <button class="tab" type="button" data-view="speakerView">说话人</button>
            <button class="tab" type="button" data-view="llmView">LLM</button>
            <button class="tab" type="button" data-view="summaryView">摘要</button>
          </div>
          <div class="view active" id="subtitleView"><pre id="subtitleOut">上传音频后，这里会显示按时间戳排列的字幕。</pre></div>
          <div class="view" id="speakerView"><pre id="speakerOut">这里会按说话人分别汇总所有内容。</pre></div>
          <div class="view" id="llmView">
            <div class="llm-brief">
              <div class="brief-card"><span>幻觉风险</span><strong id="llmRisk">-</strong></div>
              <div class="brief-card"><span>关键优化</span><strong id="llmChanges">-</strong></div>
              <div class="brief-card"><span>疑点数量</span><strong id="llmUncertain">-</strong></div>
            </div>
            <pre id="llmOut">启用 LLM/RAG 后，这里会显示清理后的结果。</pre>
          </div>
          <div class="view" id="summaryView"><pre id="summaryOut">{}</pre></div>
        </div>
      </section>
    </div>
  </main>

  <script>
    const $ = (id) => document.getElementById(id);
    const status = $("status");
    const form = $("runForm");
    const submitButton = $("submitButton");

    function fillSelect(id, values, fallback) {
      const el = $(id);
      el.innerHTML = "";
      values.forEach((value) => {
        const option = document.createElement("option");
        option.value = value;
        option.textContent = value;
        el.appendChild(option);
      });
      if (values.includes(fallback)) el.value = fallback;
    }
    function setText(id, value) { $(id).textContent = value || "-"; }
    function setBusy(value) {
      submitButton.disabled = value;
      $("resultPanel").classList.toggle("loading", value);
      if (!value) return;
      status.textContent = "Processing...";
      setText("mPipeline", "-");
      setText("mOvr", "-");
      setText("mSpeakers", "-");
      $("mStatus").className = "";
      $("mStatus").textContent = "Running";
      $("subtitleOut").textContent = "";
      $("speakerOut").textContent = "";
      $("summaryOut").textContent = "";
      renderLlm(null, "");
      renderDownloads([]);
      renderAudio([]);
    }
    function renderDownloads(items) {
      const box = $("downloads");
      box.innerHTML = "";
      (items || []).forEach((item) => {
        const a = document.createElement("a");
        a.className = "download";
        a.href = item.url;
        a.textContent = "下载 " + item.label;
        box.appendChild(a);
      });
    }
    function renderAudio(items) {
      const box = $("audioList");
      box.innerHTML = "";
      (items || []).forEach((item) => {
        const card = document.createElement("div");
        card.className = "audio-card";
        card.innerHTML = `<strong>${item.label}</strong><audio controls src="${item.url}"></audio><a class="download" href="${item.url}">下载音频</a>`;
        box.appendChild(card);
      });
    }
    function renderLlm(llm, fallbackText) {
      const data = llm || {};
      const changes = data.changes || [];
      const uncertain = data.uncertain_spans || [];
      setText("llmRisk", data.hallucination_risk || "-");
      setText("llmChanges", changes.length ? `${changes.length} 项` : "-");
      setText("llmUncertain", uncertain.length ? `${uncertain.length} 处` : "0");
      $("llmOut").textContent = data.display_text || fallbackText || "未启用 LLM/RAG 或无输出。";
    }
    function activateTab(viewId) {
      document.querySelectorAll(".tab").forEach((tab) => tab.classList.toggle("active", tab.dataset.view === viewId));
      document.querySelectorAll(".view").forEach((view) => view.classList.toggle("active", view.id === viewId));
    }
    document.querySelectorAll(".tab").forEach((tab) => tab.addEventListener("click", () => activateTab(tab.dataset.view)));

    async function loadOptions() {
      const response = await fetch("/api/options");
      const options = await response.json();
      fillSelect("asr_model", options.asr_models, options.defaults.asr);
      fillSelect("diarization_model", options.diarization_models, options.defaults.diarization);
      fillSelect("separation_model", options.separation_models, options.defaults.separation);
      fillSelect("llm_model", options.llm_models, options.defaults.llm);
      const ragBox = $("rag_domains");
      ragBox.innerHTML = "";
      options.rag_domains.forEach(([label, value]) => {
        const wrapper = document.createElement("label");
        wrapper.className = "check";
        wrapper.innerHTML = `<input type="checkbox" name="rag_domain" value="${value}" ${value === "domain:product_debate" ? "checked" : ""}/> ${label}`;
        ragBox.appendChild(wrapper);
      });
    }

    form.addEventListener("submit", async (event) => {
      event.preventDefault();
      setBusy(true);
      const data = new FormData(form);
      const domains = [...document.querySelectorAll("input[name='rag_domain']:checked")].map((item) => item.value);
      data.set("rag_domains", domains.join(","));
      try {
        const response = await fetch("/api/transcribe", { method: "POST", body: data });
        const result = await response.json();
        if (!response.ok) throw new Error(result.detail || "Request failed");
        setBusy(false);
        const summary = result.summary || {};
        setText("mPipeline", summary.selected_pipeline || "未选择");
        const ovrLabel = summary.overlap_label || summary.overlap_level || "-";
        setText("mOvr", summary.overlap_ratio == null ? ovrLabel : `${ovrLabel} / ${summary.overlap_ratio}`);
        setText("mSpeakers", summary.speaker_count ? `${summary.speaker_count}` : "-");
        $("mStatus").textContent = result.ok ? "Complete" : "Needs attention";
        $("mStatus").className = result.ok ? "" : "error";
        $("subtitleOut").textContent = result.subtitle || "";
        $("speakerOut").textContent = result.speaker_text || "";
        renderLlm(result.llm, result.llm_text);
        $("summaryOut").textContent = JSON.stringify(summary, null, 2);
        renderDownloads(result.downloads);
        renderAudio(result.separated_audio);
        activateTab("subtitleView");
        status.textContent = result.ok ? "Complete" : "Needs attention";
      } catch (error) {
        setBusy(false);
        status.textContent = "Error";
        $("mStatus").textContent = "Error";
        $("summaryOut").textContent = String(error);
        activateTab("summaryView");
      } finally {
        setBusy(false);
      }
    });

    loadOptions().catch((error) => {
      status.textContent = "Options failed";
      $("summaryOut").textContent = String(error);
    });
  </script>
</body>
</html>
"""


def _is_path_allowed(path: Path) -> bool:
    try:
        path.resolve().relative_to(ROOT.resolve())
        return True
    except ValueError:
        return False


def _safe_upload_name(filename: str | None) -> str:
    name = Path(filename or "upload.wav").name
    safe = "".join(char if char.isalnum() or char in {".", "_", "-"} else "_" for char in name)
    return safe or "upload.wav"


def build_web_app():
    if FastAPI is None:
        raise RuntimeError("FastAPI frontend requires fastapi and uvicorn. Run: pip install fastapi uvicorn python-multipart")
    app = FastAPI(title="Overlap ASR Transcript Workbench")

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return WEB_INDEX_HTML

    @app.get("/api/options")
    def options() -> dict[str, Any]:
        asr_models = _ui_model_choices(_available_asr_choices())
        diarization_models = _ui_model_choices(_available_diarization_choices())
        separation_models = _ui_model_choices(_available_separation_choices())
        llm_models = _ui_model_choices(_available_llm_choices())
        return {
            "overlap": OVERLAP_CHOICES,
            "pipelines": PIPELINE_CHOICES,
            "asr_models": asr_models,
            "diarization_models": diarization_models,
            "separation_models": separation_models,
            "llm_models": llm_models,
            "rag_domains": RAG_DOMAIN_CHOICES,
            "defaults": {
                "asr": _default_choice(asr_models, "faster-whisper:large-v3"),
                "diarization": _default_choice(
                    diarization_models,
                    "pyannote:pyannote/speaker-diarization-community-1",
                ),
                "separation": _default_choice(separation_models, "clearvoice:MossFormer2_SS_16K"),
                "llm": _default_choice(llm_models, "api"),
            },
        }

    @app.get("/api/download")
    def download(path: str):
        target = Path(path).resolve()
        if not _is_path_allowed(target) or not target.exists() or not target.is_file():
            raise HTTPException(status_code=404, detail="File not found")
        return FileResponse(str(target), filename=target.name)

    @app.post("/api/transcribe")
    async def transcribe(
        audio: UploadFile = File(...),
        overlap_level: str = Form("auto"),
        pipeline: str = Form("auto"),
        speakers: int = Form(2),
        language: str = Form("zh"),
        asr_model: str = Form("faster-whisper:large-v3"),
        diarization_model: str = Form("pyannote:pyannote/speaker-diarization-community-1"),
        separation_model: str = Form("clearvoice:MossFormer2_SS_16K"),
        llm_model: str = Form("api"),
        hotwords: str = Form(""),
        rag_domains: str = Form("domain:product_debate"),
        llm_extra_prompt: str = Form(""),
        use_llm: bool = Form(False),
        mock: bool = Form(False),
    ) -> dict[str, Any]:
        run_id = time.strftime("%Y%m%d_%H%M%S")
        upload_dir = ROOT / "outputs" / "speaker_app" / "uploads" / run_id
        upload_dir.mkdir(parents=True, exist_ok=True)
        upload_path = upload_dir / _safe_upload_name(audio.filename)
        with upload_path.open("wb") as handle:
            shutil.copyfileobj(audio.file, handle)
        domains = [item for item in rag_domains.split(",") if item]
        return run_transcription(
            upload_path,
            overlap_level,
            pipeline,
            speakers,
            language,
            asr_model,
            diarization_model,
            separation_model,
            hotwords,
            domains,
            llm_extra_prompt,
            use_llm,
            llm_model,
            mock,
            current_upload_dir=upload_dir,
        )

    return app


def build_app():
    import gradio as gr

    if not hasattr(gr, "Blocks"):
        raise RuntimeError("This app requires gradio>=4.0.0. Run: pip install -U gradio")
    asr_choices = _ui_model_choices(_available_asr_choices())
    diarization_choices = _ui_model_choices(_available_diarization_choices())
    separation_choices = _ui_model_choices(_available_separation_choices())
    llm_choices = _ui_model_choices(_available_llm_choices())

    with gr.Blocks(title="Overlap ASR Speaker Transcript") as app:
        with gr.Column(elem_classes=["app-shell"]):
            gr.Markdown("# Overlap ASR Transcript Workbench")
            with gr.Row():
                with gr.Column(scale=1):
                    audio_file = gr.Audio(label="上传音频", type="filepath")
                    run_button = gr.Button("开始转写", variant="primary")
                    overlap_level = gr.State("auto")
                    pipeline_choice = gr.State("auto")
                    speakers = gr.State(2)
                    mock = gr.State(False)
                    language = gr.Dropdown(
                        label="语言",
                        choices=["zh", "en", "ja", "ko", "auto"],
                        value="zh",
                    )
                    with gr.Accordion("Models", open=True):
                        asr_model = gr.Dropdown(
                            label="ASR",
                            choices=asr_choices,
                            value=_default_choice(asr_choices, "faster-whisper:large-v3"),
                            allow_custom_value=False,
                        )
                        diarization_model = gr.Dropdown(
                            label="Diarization",
                            choices=diarization_choices,
                            value=_default_choice(
                                diarization_choices,
                                "pyannote:pyannote/speaker-diarization-community-1",
                            ),
                            allow_custom_value=False,
                        )
                        separation_model = gr.Dropdown(
                            label="Separation",
                            choices=separation_choices,
                            value=_default_choice(
                                separation_choices,
                                "clearvoice:MossFormer2_SS_16K",
                            ),
                            allow_custom_value=False,
                        )
                        llm_model = gr.Dropdown(
                            label="LLM",
                            choices=llm_choices,
                            value=_default_choice(llm_choices, "api"),
                            allow_custom_value=False,
                        )
                    with gr.Accordion("高级 RAG 和提示词", open=False):
                        rag_domains = gr.CheckboxGroup(
                            label="RAG 领域",
                            choices=RAG_DOMAIN_CHOICES,
                            value=["domain:product_debate"],
                        )
                        hotwords = gr.Textbox(label="ASR 热词 / Prompt", value="", lines=3)
                        llm_extra_prompt = gr.Textbox(
                            label="LLM 补充提示词",
                            value="",
                            lines=4,
                        )
                        use_llm = gr.Checkbox(label="LLM/RAG cleanup", value=False)

                with gr.Column(scale=2):
                    with gr.Tabs():
                        with gr.Tab("字幕"):
                            subtitle_output = gr.Textbox(label="按时间戳字幕", lines=22)
                            subtitle_file = gr.File(label="字幕文件")
                        with gr.Tab("说话人"):
                            speaker_output = gr.Textbox(label="按说话人汇总", lines=22)
                            speaker_file = gr.File(label="说话人文本")
                        with gr.Tab("LLM/RAG"):
                            llm_output = gr.Textbox(label="LLM 清理结果", lines=22)
                        with gr.Tab("运行摘要"):
                            summary_output = gr.Textbox(label="Pipeline 选择", lines=22)

            run_button.click(
                process_audio,
                inputs=[
                    audio_file,
                    overlap_level,
                    pipeline_choice,
                    speakers,
                    language,
                    asr_model,
                    diarization_model,
                    separation_model,
                    hotwords,
                    rag_domains,
                    llm_extra_prompt,
                    use_llm,
                    llm_model,
                    mock,
                ],
                outputs=[
                    summary_output,
                    subtitle_output,
                    speaker_output,
                    llm_output,
                    subtitle_file,
                    speaker_file,
                ],
            )
    return app


def main() -> int:
    parser = argparse.ArgumentParser(description="Launch speaker transcript app.")
    parser.add_argument("--port", type=int, default=7861)
    parser.add_argument(
        "--port-retries",
        type=int,
        default=20,
        help="Try this many following ports if the requested port is busy.",
    )
    parser.add_argument(
        "--env-file",
        default=str(DEFAULT_ENV_FILE),
        help="Path to the .env file used by the frontend and model providers.",
    )
    parser.add_argument(
        "--ui",
        choices=["web", "gradio"],
        default="web",
        help="Frontend implementation to launch. The default web UI does not use Gradio.",
    )
    parser.add_argument("--host", default=os.environ.get("GRADIO_SERVER_NAME", "127.0.0.1"))
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()

    env_path = _load_frontend_env(args.env_file)
    print(_env_loaded_summary(env_path))

    if args.ui == "gradio":
        app = build_app()
    else:
        app = build_web_app()
    last_error = None
    for offset in range(max(0, args.port_retries) + 1):
        port = args.port + offset
        try:
            if offset:
                print(f"Port {args.port} is busy; trying {port}...")
            if args.ui == "gradio":
                _launch_app(app, host=args.host, port=port, share=args.share)
            else:
                import uvicorn

                uvicorn.run(app, host=args.host, port=port)
            return 0
        except OSError as exc:
            last_error = exc
            if "Cannot find empty port" not in str(exc):
                raise
    if last_error is not None:
        raise last_error
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
