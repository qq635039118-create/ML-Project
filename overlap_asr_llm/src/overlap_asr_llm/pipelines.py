"""Experiment pipeline orchestration."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
import tempfile
import time

from .config import ExperimentConfig, Sample
from .metrics import cer, wer
from .providers import make_asr, make_diarizer, make_llm_refiner, make_separator


class ProviderCache:
    def __init__(self, config: ExperimentConfig) -> None:
        self.config = config
        self._asr = None
        self._diarizer = None
        self._separator = None
        self._refiner = None

    def asr(self):
        if self._asr is None:
            self._asr = make_asr(self.config.models.get("asr", "mock"))
        return self._asr

    def diarizer(self):
        if self._diarizer is None:
            self._diarizer = make_diarizer(self.config.models.get("diarization", "mock"))
        return self._diarizer

    def separator(self):
        if self._separator is None:
            self._separator = make_separator(self.config.models.get("separation", "mock"))
        return self._separator

    def refiner(self):
        if self._refiner is None:
            self._refiner = make_llm_refiner(self.config.models.get("llm", "mock"))
        return self._refiner


@dataclass
class PipelineResult:
    sample_id: str
    audio_path: str
    overlap_level: str
    pipeline: str
    model: str
    text: str
    speaker_labels: str
    runtime_seconds: float
    cer: float | None
    wer: float | None
    text_cer: float | None = None
    text_wer: float | None = None
    error: str | None = None
    segments: list[dict[str, object]] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _score(reference: str | None, text: str) -> tuple[float | None, float | None]:
    if not reference:
        return None, None
    return cer(reference, text), wer(reference, text)


def _labels_from_segments(segments: list[dict[str, object]]) -> str:
    labels = [str(segment.get("speaker", "UNKNOWN")) for segment in segments]
    return ",".join(dict.fromkeys(labels))


def _format_timestamp(seconds: float) -> str:
    seconds = max(seconds, 0.0)
    minutes, second_value = divmod(seconds, 60)
    hours, minute_value = divmod(int(minutes), 60)
    return f"{hours:02d}:{minute_value:02d}:{second_value:06.3f}"


def _subtitle_text(segments: list[dict[str, object]]) -> str:
    lines = []
    for segment in segments:
        start = _format_timestamp(float(segment.get("start", 0.0)))
        end = _format_timestamp(float(segment.get("end", 0.0)))
        speaker = str(segment.get("speaker", "UNKNOWN"))
        text = str(segment.get("text", "")).strip()
        lines.append(f"{start} --> {end} [{speaker}] {text}".strip())
    return "\n".join(lines)


def _plain_text_from_segments(segments: list[dict[str, object]]) -> str:
    return " ".join(str(segment.get("text", "")).strip() for segment in segments).strip()


def _write_audio_excerpt(
    audio_path: Path,
    start: float,
    end: float,
    output_path: Path,
) -> None:
    import soundfile as sf

    with sf.SoundFile(str(audio_path)) as source:
        sample_rate = source.samplerate
        total_frames = len(source)
        start_frame = max(0, min(total_frames, int(start * sample_rate)))
        end_frame = max(0, min(total_frames, int(end * sample_rate)))
        if end_frame <= start_frame:
            end_frame = min(total_frames, start_frame + max(1, int(0.1 * sample_rate)))
        source.seek(start_frame)
        audio = source.read(end_frame - start_frame, dtype="float32", always_2d=True)
    sf.write(str(output_path), audio, sample_rate)


def _transcribe_speaker_turns(
    asr,
    audio_path: Path,
    speaker_turns: list[dict[str, object]],
    language: str,
    prompt: str | None = None,
) -> list[dict[str, object]]:
    ordered_turns = sorted(speaker_turns, key=lambda segment: float(segment.get("start", 0.0)))
    if getattr(asr, "name", "") == "mock_asr":
        transcript = asr.transcribe(audio_path, language, prompt=prompt)
        base_text = transcript.text.strip()
        return [
            {
                "start": float(turn.get("start", 0.0)),
                "end": float(turn.get("end", turn.get("start", 0.0))),
                "speaker": str(turn.get("speaker", "UNKNOWN")),
                "text": str(turn.get("text", "")).strip() or f"{base_text} turn {index}",
            }
            for index, turn in enumerate(ordered_turns, start=1)
        ]

    segments: list[dict[str, object]] = []
    with tempfile.TemporaryDirectory(prefix="overlap_asr_llm_diarization_") as tmpdir:
        tmp_path = Path(tmpdir)
        for index, turn in enumerate(ordered_turns, start=1):
            start = float(turn.get("start", 0.0))
            end = float(turn.get("end", start))
            excerpt_path = tmp_path / f"turn_{index:04d}.wav"
            _write_audio_excerpt(audio_path, start, end, excerpt_path)
            transcript = asr.transcribe(excerpt_path, language, prompt=prompt)
            text = transcript.text.strip()
            if not text:
                text = " ".join(
                    str(segment.get("text", "")).strip()
                    for segment in transcript.segments
                    if str(segment.get("text", "")).strip()
                )
            segments.append(
                {
                    "start": start,
                    "end": end,
                    "speaker": str(turn.get("speaker", "UNKNOWN")),
                    "text": text,
                }
            )
    return segments


def _error_result(
    sample: Sample,
    pipeline: str,
    model: str,
    started: float,
    error: Exception,
) -> PipelineResult:
    return PipelineResult(
        sample_id=sample.id,
        audio_path=str(sample.audio_path),
        overlap_level=sample.overlap_level,
        pipeline=pipeline,
        model=model,
        text="",
        speaker_labels="",
        runtime_seconds=round(time.perf_counter() - started, 4),
        cer=None,
        wer=None,
        text_cer=None,
        text_wer=None,
        error=f"{type(error).__name__}: {error}",
        segments=[],
    )


def run_direct_asr(
    config: ExperimentConfig,
    sample: Sample,
    providers: ProviderCache | None = None,
) -> PipelineResult:
    started = time.perf_counter()
    model_name = config.models.get("asr", "mock")
    try:
        asr = providers.asr() if providers else make_asr(model_name)
        started = time.perf_counter()
        transcript = asr.transcribe(
            sample.audio_path,
            config.language,
            prompt=config.asr_prompt,
        )
        cer_value, wer_value = _score(sample.reference, transcript.text)
        return PipelineResult(
            sample_id=sample.id,
            audio_path=str(sample.audio_path),
            overlap_level=sample.overlap_level,
            pipeline="direct_asr",
            model=asr.name,
            text=transcript.text,
            speaker_labels=_labels_from_segments(transcript.segments),
            runtime_seconds=round(time.perf_counter() - started, 4),
            cer=cer_value,
            wer=wer_value,
            segments=transcript.segments,
        )
    except Exception as exc:
        return _error_result(sample, "direct_asr", model_name, started, exc)


def run_diarization_asr(
    config: ExperimentConfig,
    sample: Sample,
    providers: ProviderCache | None = None,
) -> PipelineResult:
    started = time.perf_counter()
    asr_name = config.models.get("asr", "mock")
    diarizer_name = config.models.get("diarization", "mock")
    model_name = f"{asr_name}+{diarizer_name}"
    try:
        diarizer = providers.diarizer() if providers else make_diarizer(diarizer_name)
        started = time.perf_counter()
        speaker_turns = diarizer.diarize(sample.audio_path, sample.speakers)
        if hasattr(diarizer, "release_gpu"):
            diarizer.release_gpu()
        asr = providers.asr() if providers else make_asr(asr_name)
        segments = _transcribe_speaker_turns(
            asr,
            sample.audio_path,
            speaker_turns,
            config.language,
            prompt=config.asr_prompt,
        )
        if hasattr(asr, "release_gpu"):
            asr.release_gpu()
        subtitle_text = _subtitle_text(segments)
        plain_text = _plain_text_from_segments(segments)
        cer_value, wer_value = _score(sample.reference, subtitle_text)
        text_cer_value, text_wer_value = _score(sample.reference, plain_text)
        return PipelineResult(
            sample_id=sample.id,
            audio_path=str(sample.audio_path),
            overlap_level=sample.overlap_level,
            pipeline="diarization_asr",
            model=f"{asr.name}+{diarizer.name}",
            text=subtitle_text,
            speaker_labels=_labels_from_segments(segments),
            runtime_seconds=round(time.perf_counter() - started, 4),
            cer=cer_value,
            wer=wer_value,
            text_cer=text_cer_value,
            text_wer=text_wer_value,
            segments=segments,
        )
    except Exception as exc:
        return _error_result(sample, "diarization_asr", model_name, started, exc)


def run_separation_asr(
    config: ExperimentConfig,
    sample: Sample,
    providers: ProviderCache | None = None,
) -> PipelineResult:
    started = time.perf_counter()
    asr_name = config.models.get("asr", "mock")
    separator_name = config.models.get("separation", "mock")
    model_name = f"{separator_name}+{asr_name}"
    try:
        asr = providers.asr() if providers else make_asr(asr_name)
        separator = providers.separator() if providers else make_separator(separator_name)
        started = time.perf_counter()
        separated_dir = config.output_dir / "separated_audio" / sample.id
        speaker_paths = separator.separate(sample.audio_path, separated_dir, sample.speakers)
        parts = []
        segments = []
        for index, speaker_path in enumerate(speaker_paths, start=1):
            transcript = asr.transcribe(
                speaker_path,
                config.language,
                prompt=config.asr_prompt,
            )
            speaker = f"SPEAKER{index}"
            parts.append(f"[{speaker}] {transcript.text}")
            transcript_segments = transcript.segments or [
                {
                    "start": 0.0,
                    "end": 0.0,
                    "text": transcript.text,
                    "speaker": "UNKNOWN",
                }
            ]
            segments.extend(
                {
                    **segment,
                    "speaker": speaker,
                    "source_index": index,
                    "separated_audio_path": str(speaker_path),
                }
                for segment in transcript_segments
            )
        text = " ".join(parts)
        plain_text = _plain_text_from_segments(segments)
        cer_value, wer_value = _score(sample.reference, plain_text)
        return PipelineResult(
            sample_id=sample.id,
            audio_path=str(sample.audio_path),
            overlap_level=sample.overlap_level,
            pipeline="separation_asr",
            model=f"{separator.name}+{asr.name}",
            text=text,
            speaker_labels=_labels_from_segments(segments),
            runtime_seconds=round(time.perf_counter() - started, 4),
            cer=cer_value,
            wer=wer_value,
            segments=segments,
        )
    except Exception as exc:
        return _error_result(sample, "separation_asr", model_name, started, exc)


def run_llm_rag_refine(
    config: ExperimentConfig,
    sample: Sample,
    source_results: list[PipelineResult],
    providers: ProviderCache | None = None,
) -> PipelineResult:
    started = time.perf_counter()
    llm_name = config.models.get("llm", "mock")
    try:
        refiner = providers.refiner() if providers else make_llm_refiner(llm_name)
        started = time.perf_counter()
        source_text = "\n".join(
            f"{result.pipeline}: {result.text}"
            for result in source_results
            if result.sample_id == sample.id and not result.error
        )
        refined = refiner.refine(source_text, config.rag_context)
        cer_value, wer_value = _score(sample.reference, refined)
        return PipelineResult(
            sample_id=sample.id,
            audio_path=str(sample.audio_path),
            overlap_level=sample.overlap_level,
            pipeline="llm_rag_refine",
            model=refiner.name,
            text=refined,
            speaker_labels="LLM_REFINED",
            runtime_seconds=round(time.perf_counter() - started, 4),
            cer=cer_value,
            wer=wer_value,
            segments=[],
        )
    except Exception as exc:
        return _error_result(sample, "llm_rag_refine", llm_name, started, exc)


def run_all(config: ExperimentConfig) -> list[PipelineResult]:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    providers = ProviderCache(config)
    results: list[PipelineResult] = []
    for sample in config.samples:
        if "direct_asr" in config.pipelines:
            results.append(run_direct_asr(config, sample, providers))
        if "diarization_asr" in config.pipelines:
            results.append(run_diarization_asr(config, sample, providers))
        if "separation_asr" in config.pipelines:
            results.append(run_separation_asr(config, sample, providers))
        if "llm_rag_refine" in config.pipelines:
            results.append(run_llm_rag_refine(config, sample, results, providers))
    return results
