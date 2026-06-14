"""Experiment pipeline orchestration."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
import tempfile
import time

from .config import ExperimentConfig, Sample
from .metrics import cer, speaker_block_score, wer
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
    flat_cer: float | None = None
    flat_wer: float | None = None
    timeline_cer: float | None = None
    timeline_wer: float | None = None
    speaker_block_cer: float | None = None
    speaker_block_wer: float | None = None
    best_speaker_mapping: str = ""
    score_basis: str = "flat"
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


def _timeline_text_from_segments(segments: list[dict[str, object]]) -> str:
    ordered = sorted(
        segments,
        key=lambda segment: (
            float(segment.get("start", 0.0)),
            float(segment.get("end", 0.0)),
            str(segment.get("speaker", "")),
        ),
    )
    return _plain_text_from_segments(ordered)


def _speaker_texts_from_segments(
    segments: list[dict[str, object]],
) -> dict[str, str]:
    speaker_texts: dict[str, list[str]] = {}
    for segment in segments:
        text = str(segment.get("text", "")).strip()
        if not text:
            continue
        speaker = str(segment.get("speaker", "UNKNOWN"))
        speaker_texts.setdefault(speaker, []).append(text)
    return {
        speaker: " ".join(parts).strip()
        for speaker, parts in speaker_texts.items()
        if " ".join(parts).strip()
    }


@dataclass(frozen=True)
class ScoreBundle:
    cer: float | None
    wer: float | None
    text_cer: float | None
    text_wer: float | None
    flat_cer: float | None
    flat_wer: float | None
    timeline_cer: float | None
    timeline_wer: float | None
    speaker_block_cer: float | None
    speaker_block_wer: float | None
    best_speaker_mapping: str
    score_basis: str


def _score_bundle(
    sample: Sample,
    flat_text: str,
    segments: list[dict[str, object]] | None = None,
    prefer_speaker_block: bool = False,
) -> ScoreBundle:
    flat_cer, flat_wer = _score(sample.reference, flat_text)
    timeline_text = _timeline_text_from_segments(segments or []) if segments else flat_text
    timeline_cer, timeline_wer = _score(sample.reference, timeline_text)

    speaker_cer = None
    speaker_wer = None
    best_mapping = ""
    if sample.reference_speakers and segments:
        reference_speakers = {
            item.speaker: item.text for item in sample.reference_speakers
        }
        hypothesis_speakers = _speaker_texts_from_segments(segments)
        if len(hypothesis_speakers) >= 2:
            speaker_score = speaker_block_score(
                reference_speakers,
                hypothesis_speakers,
            )
            if speaker_score is not None:
                speaker_cer = speaker_score.cer
                speaker_wer = speaker_score.wer
                best_mapping = json.dumps(
                    speaker_score.mapping,
                    ensure_ascii=False,
                    sort_keys=True,
                )

    if prefer_speaker_block and speaker_cer is not None and speaker_wer is not None:
        primary_cer = speaker_cer
        primary_wer = speaker_wer
        score_basis = "speaker_block"
    elif timeline_cer is not None and timeline_wer is not None:
        primary_cer = timeline_cer
        primary_wer = timeline_wer
        score_basis = "timeline"
    else:
        primary_cer = flat_cer
        primary_wer = flat_wer
        score_basis = "flat"

    return ScoreBundle(
        cer=primary_cer,
        wer=primary_wer,
        text_cer=timeline_cer,
        text_wer=timeline_wer,
        flat_cer=flat_cer,
        flat_wer=flat_wer,
        timeline_cer=timeline_cer,
        timeline_wer=timeline_wer,
        speaker_block_cer=speaker_cer,
        speaker_block_wer=speaker_wer,
        best_speaker_mapping=best_mapping,
        score_basis=score_basis,
    )


def _result_scores(score: ScoreBundle) -> dict[str, object]:
    return {
        "cer": score.cer,
        "wer": score.wer,
        "text_cer": score.text_cer,
        "text_wer": score.text_wer,
        "flat_cer": score.flat_cer,
        "flat_wer": score.flat_wer,
        "timeline_cer": score.timeline_cer,
        "timeline_wer": score.timeline_wer,
        "speaker_block_cer": score.speaker_block_cer,
        "speaker_block_wer": score.speaker_block_wer,
        "best_speaker_mapping": score.best_speaker_mapping,
        "score_basis": score.score_basis,
    }


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
        flat_cer=None,
        flat_wer=None,
        timeline_cer=None,
        timeline_wer=None,
        speaker_block_cer=None,
        speaker_block_wer=None,
        best_speaker_mapping="",
        score_basis="error",
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
        score = _score_bundle(
            sample,
            transcript.text,
            transcript.segments,
            prefer_speaker_block=False,
        )
        return PipelineResult(
            sample_id=sample.id,
            audio_path=str(sample.audio_path),
            overlap_level=sample.overlap_level,
            pipeline="direct_asr",
            model=asr.name,
            text=transcript.text,
            speaker_labels=_labels_from_segments(transcript.segments),
            runtime_seconds=round(time.perf_counter() - started, 4),
            **_result_scores(score),
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
        score = _score_bundle(
            sample,
            plain_text,
            segments,
            prefer_speaker_block=True,
        )
        return PipelineResult(
            sample_id=sample.id,
            audio_path=str(sample.audio_path),
            overlap_level=sample.overlap_level,
            pipeline="diarization_asr",
            model=f"{asr.name}+{diarizer.name}",
            text=subtitle_text,
            speaker_labels=_labels_from_segments(segments),
            runtime_seconds=round(time.perf_counter() - started, 4),
            **_result_scores(score),
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
        score = _score_bundle(
            sample,
            plain_text,
            segments,
            prefer_speaker_block=True,
        )
        return PipelineResult(
            sample_id=sample.id,
            audio_path=str(sample.audio_path),
            overlap_level=sample.overlap_level,
            pipeline="separation_asr",
            model=f"{separator.name}+{asr.name}",
            text=text,
            speaker_labels=_labels_from_segments(segments),
            runtime_seconds=round(time.perf_counter() - started, 4),
            **_result_scores(score),
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
        score = _score_bundle(sample, refined)
        return PipelineResult(
            sample_id=sample.id,
            audio_path=str(sample.audio_path),
            overlap_level=sample.overlap_level,
            pipeline="llm_rag_refine",
            model=refiner.name,
            text=refined,
            speaker_labels="LLM_REFINED",
            runtime_seconds=round(time.perf_counter() - started, 4),
            **_result_scores(score),
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
