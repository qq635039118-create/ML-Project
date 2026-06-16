"""Post-run readability evaluation for ASR experiment outputs."""

from __future__ import annotations

import csv
from dataclasses import dataclass
import json
import math
from pathlib import Path
import re
from typing import Any, Protocol

from .config import ExperimentConfig, Sample


DEFAULT_BERT_MODEL = "bert-base-chinese"
READABILITY_FIELDNAMES = [
    "sample_id",
    "overlap_level",
    "ovr",
    "ovr_source",
    "pipeline",
    "model",
    "cer",
    "wer",
    "bert_precision",
    "bert_recall",
    "bert_f1",
    "bert_f2",
    "speaker_block_cer",
    "speaker_consistency",
    "trs_text",
    "trs_speaker",
    "runtime_seconds",
    "error",
]

_SUBTITLE_PREFIX_RE = re.compile(
    r"^\s*\d{2}:\d{2}:\d{2}(?:[.,]\d+)?\s*-->\s*"
    r"\d{2}:\d{2}:\d{2}(?:[.,]\d+)?\s*"
)
_SPEAKER_PREFIX_RE = re.compile(r"^\s*\[[^\]]+\]\s*")


class BertScorer(Protocol):
    def __call__(
        self,
        hypotheses: list[str],
        references: list[str],
        *,
        model_type: str,
        device: str,
        batch_size: int,
    ) -> list[tuple[float, float, float]]:
        """Return BERTScore precision, recall, and F1 triples."""


@dataclass(frozen=True)
class ReadabilityEvaluation:
    rows: list[dict[str, object]]
    output_dir: Path


def compute_overlap_ratio(segments: list[dict[str, object]]) -> float | None:
    """Compute overlapped speech ratio from timestamped speaker segments.

    OVR = duration with two or more active speakers / duration with speech.
    """

    events: list[tuple[float, int]] = []
    for segment in segments:
        start = _coerce_float(segment.get("start"))
        end = _coerce_float(segment.get("end"))
        if start is None or end is None or end <= start:
            continue
        events.append((start, 1))
        events.append((end, -1))

    if not events:
        return None

    events.sort(key=lambda item: (item[0], item[1]))
    active = 0
    previous_time: float | None = None
    speech_duration = 0.0
    overlap_duration = 0.0

    for timestamp, delta in events:
        if previous_time is not None and timestamp > previous_time:
            duration = timestamp - previous_time
            if active > 0:
                speech_duration += duration
            if active >= 2:
                overlap_duration += duration
        active += delta
        previous_time = timestamp

    if speech_duration <= 0:
        return None
    return overlap_duration / speech_duration


def bert_f_beta(precision: float | None, recall: float | None, beta: float = 2.0) -> float | None:
    if precision is None or recall is None or beta <= 0:
        return None
    denominator = (beta * beta * precision) + recall
    if denominator <= 0:
        return 0.0
    return ((1 + beta * beta) * precision * recall) / denominator


def speaker_consistency(speaker_block_cer: float | None) -> float | None:
    if speaker_block_cer is None:
        return None
    return 1 - min(max(speaker_block_cer, 0.0), 1.0)


def trs_text(cer: float | None, bert_f2: float | None) -> float | None:
    if cer is None or bert_f2 is None:
        return None
    character_accuracy = 1 - min(max(cer, 0.0), 1.0)
    return 100 * math.sqrt(character_accuracy * max(bert_f2, 0.0))


def trs_speaker(
    cer: float | None,
    bert_f2: float | None,
    speaker_block_cer: float | None,
) -> float | None:
    consistency = speaker_consistency(speaker_block_cer)
    if cer is None or bert_f2 is None or consistency is None:
        return None
    character_accuracy = 1 - min(max(cer, 0.0), 1.0)
    return 100 * ((character_accuracy * max(bert_f2, 0.0) * consistency) ** (1 / 3))


def evaluate_results(
    config: ExperimentConfig,
    results_path: str | Path,
    device: str = "auto",
    bert_model: str = DEFAULT_BERT_MODEL,
    batch_size: int = 1,
    scorer: BertScorer | None = None,
) -> ReadabilityEvaluation:
    results_path = Path(results_path)
    rows = _load_result_rows(results_path)
    samples_by_id = {sample.id: sample for sample in config.samples}
    ovr_by_sample = _sample_overlap_ratios(rows)

    scoring_jobs: list[tuple[int, str, str]] = []
    output_rows: list[dict[str, object]] = []
    for row in rows:
        sample_id = str(row.get("sample_id", ""))
        sample = samples_by_id.get(sample_id)
        reference = sample.reference if sample else None
        text = _text_for_semantic_scoring(row)
        ovr, ovr_source = ovr_by_sample.get(sample_id, (None, "unavailable"))
        bert_precision = bert_recall = bert_f1 = bert_f2_value = None

        output_row: dict[str, object] = {
            "sample_id": sample_id,
            "overlap_level": row.get("overlap_level", sample.overlap_level if sample else ""),
            "ovr": ovr,
            "ovr_source": ovr_source,
            "pipeline": row.get("pipeline", ""),
            "model": row.get("model", ""),
            "cer": _coerce_float(row.get("cer")),
            "wer": _coerce_float(row.get("wer")),
            "bert_precision": bert_precision,
            "bert_recall": bert_recall,
            "bert_f1": bert_f1,
            "bert_f2": bert_f2_value,
            "speaker_block_cer": _coerce_float(row.get("speaker_block_cer")),
            "speaker_consistency": speaker_consistency(
                _coerce_float(row.get("speaker_block_cer"))
            ),
            "trs_text": None,
            "trs_speaker": None,
            "runtime_seconds": _coerce_float(row.get("runtime_seconds")),
            "error": row.get("error") or "",
        }
        output_rows.append(output_row)

        if reference and text and not output_row["error"]:
            scoring_jobs.append((len(output_rows) - 1, text, reference))

    if scoring_jobs:
        resolved_device = _resolve_device(device)
        score_fn = scorer or _bert_score
        bert_scores = score_fn(
            [job[1] for job in scoring_jobs],
            [job[2] for job in scoring_jobs],
            model_type=bert_model,
            device=resolved_device,
            batch_size=batch_size,
        )
        for (row_index, _, _), (precision, recall, f1) in zip(scoring_jobs, bert_scores):
            output_row = output_rows[row_index]
            f2 = bert_f_beta(precision, recall, beta=2.0)
            output_row["bert_precision"] = precision
            output_row["bert_recall"] = recall
            output_row["bert_f1"] = f1
            output_row["bert_f2"] = f2
            output_row["speaker_consistency"] = speaker_consistency(
                _coerce_float(output_row["speaker_block_cer"])
            )
            output_row["trs_text"] = trs_text(_coerce_float(output_row["cer"]), f2)
            output_row["trs_speaker"] = trs_speaker(
                _coerce_float(output_row["cer"]),
                f2,
                _coerce_float(output_row["speaker_block_cer"]),
            )

    output_dir = results_path.parent
    _write_readability_outputs(output_rows, output_dir)
    return ReadabilityEvaluation(rows=output_rows, output_dir=output_dir)


def _load_result_rows(results_path: Path) -> list[dict[str, object]]:
    with results_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("results JSON must contain a list of result rows.")
    return [row for row in data if isinstance(row, dict)]


def _sample_overlap_ratios(
    rows: list[dict[str, object]],
) -> dict[str, tuple[float | None, str]]:
    by_sample: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        sample_id = str(row.get("sample_id", ""))
        if sample_id:
            by_sample.setdefault(sample_id, []).append(row)

    ratios: dict[str, tuple[float | None, str]] = {}
    priority = ["diarization_asr", "diarization_turn_asr", "separation_asr"]
    for sample_id, sample_rows in by_sample.items():
        selected_ratio: float | None = None
        selected_source = "unavailable"
        for pipeline in priority:
            for row in sample_rows:
                if row.get("pipeline") != pipeline:
                    continue
                segments = row.get("segments", [])
                if not isinstance(segments, list):
                    continue
                ratio = compute_overlap_ratio(
                    [segment for segment in segments if isinstance(segment, dict)]
                )
                if ratio is not None:
                    selected_ratio = ratio
                    selected_source = "estimated"
                    break
            if selected_ratio is not None:
                break
        ratios[sample_id] = (selected_ratio, selected_source)
    return ratios


def _text_for_semantic_scoring(row: dict[str, object]) -> str:
    segments = row.get("segments", [])
    if isinstance(segments, list) and segments:
        ordered = sorted(
            (segment for segment in segments if isinstance(segment, dict)),
            key=lambda segment: (
                _coerce_float(segment.get("start")) or 0.0,
                _coerce_float(segment.get("end")) or 0.0,
                str(segment.get("speaker", "")),
            ),
        )
        text = " ".join(str(segment.get("text", "")).strip() for segment in ordered)
        if text.strip():
            return text.strip()

    raw_text = str(row.get("text", "") or "")
    try:
        data = json.loads(raw_text)
    except json.JSONDecodeError:
        return _spoken_text_from_subtitle(raw_text)
    if isinstance(data, dict):
        refined = data.get("refined_text")
        if isinstance(refined, str) and refined.strip():
            return _spoken_text_from_subtitle(refined)
    return _spoken_text_from_subtitle(raw_text)


def _spoken_text_from_subtitle(text: str) -> str:
    spoken_lines = []
    for line in text.splitlines():
        line = _SUBTITLE_PREFIX_RE.sub("", line.strip())
        line = _SPEAKER_PREFIX_RE.sub("", line).strip()
        if line:
            spoken_lines.append(line)
    return " ".join(spoken_lines).strip()


def _resolve_device(device: str) -> str:
    if device != "auto":
        return device
    try:
        import torch  # type: ignore
    except ImportError:
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def _bert_score(
    hypotheses: list[str],
    references: list[str],
    *,
    model_type: str,
    device: str,
    batch_size: int,
) -> list[tuple[float, float, float]]:
    try:
        from bert_score import score  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "BERTScore evaluation requires the optional 'bert-score' package. "
            "Install requirements.txt before running 'overlap-asr-llm evaluate'."
        ) from exc

    precision, recall, f1 = score(
        hypotheses,
        references,
        model_type=model_type,
        device=device,
        batch_size=batch_size,
        verbose=False,
        rescale_with_baseline=False,
    )
    return [
        (float(p), float(r), float(f))
        for p, r, f in zip(precision.tolist(), recall.tolist(), f1.tolist())
    ]


def _write_readability_outputs(rows: list[dict[str, object]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "readability_results.json").open("w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)

    with (output_dir / "readability_results.csv").open(
        "w",
        encoding="utf-8",
        newline="",
    ) as f:
        writer = csv.DictWriter(f, fieldnames=READABILITY_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    (output_dir / "readability_summary.md").write_text(
        "\n".join(_summary_lines(rows)) + "\n",
        encoding="utf-8",
    )


def _summary_lines(rows: list[dict[str, object]]) -> list[str]:
    lines = [
        "# Readability Evaluation Summary",
        "",
        "## Pipeline Ranking",
        "",
        "| Rank | Pipeline | Runs | Avg CER | Avg WER | Avg BERT F2 | Avg TRS Text | Avg TRS Speaker | Avg Runtime |",
        "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for rank, row in enumerate(_pipeline_ranking(rows), start=1):
        lines.append(
            "| "
            f"{rank} | "
            f"{row['pipeline']} | "
            f"{row['runs']} | "
            f"{_metric(row['avg_cer'])} | "
            f"{_metric(row['avg_wer'])} | "
            f"{_metric(row['avg_bert_f2'])} | "
            f"{_metric(row['avg_trs_text'])} | "
            f"{_metric(row['avg_trs_speaker'])} | "
            f"{_seconds(row['avg_runtime'])} |"
        )

    lines.extend(
        [
            "",
            "## Best By OVR",
            "",
            "| Sample | Overlap | OVR | Best By TRS Text | TRS Text | Best By BERT F2 | BERT F2 |",
            "| --- | --- | ---: | --- | ---: | --- | ---: |",
        ]
    )
    for row in _best_by_sample(rows):
        lines.append(
            "| "
            f"{row['sample_id']} | "
            f"{row['overlap_level']} | "
            f"{_metric(row['ovr'])} | "
            f"{row['best_trs_pipeline']} | "
            f"{_metric(row['best_trs_text'])} | "
            f"{row['best_bert_pipeline']} | "
            f"{_metric(row['best_bert_f2'])} |"
        )

    lines.extend(
        [
            "",
            "## High Overlap Review",
            "",
            "| Sample | OVR | Pipeline | BERT Precision | BERT Recall | BERT F2 | TRS Text | Notes |",
            "| --- | ---: | --- | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for row in _high_overlap_rows(rows):
        notes = _failure_notes(row)
        lines.append(
            "| "
            f"{row.get('sample_id', '')} | "
            f"{_metric(_coerce_float(row.get('ovr')))} | "
            f"{row.get('pipeline', '')} | "
            f"{_metric(_coerce_float(row.get('bert_precision')))} | "
            f"{_metric(_coerce_float(row.get('bert_recall')))} | "
            f"{_metric(_coerce_float(row.get('bert_f2')))} | "
            f"{_metric(_coerce_float(row.get('trs_text')))} | "
            f"{notes} |"
        )
    return lines


def _pipeline_ranking(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    pipelines = dict.fromkeys(str(row.get("pipeline", "")) for row in rows)
    ranking = []
    for pipeline in pipelines:
        pipeline_rows = [row for row in rows if row.get("pipeline") == pipeline]
        ranking.append(
            {
                "pipeline": pipeline,
                "runs": len(pipeline_rows),
                "avg_cer": _average(_coerce_float(row.get("cer")) for row in pipeline_rows),
                "avg_wer": _average(_coerce_float(row.get("wer")) for row in pipeline_rows),
                "avg_bert_f2": _average(
                    _coerce_float(row.get("bert_f2")) for row in pipeline_rows
                ),
                "avg_trs_text": _average(
                    _coerce_float(row.get("trs_text")) for row in pipeline_rows
                ),
                "avg_trs_speaker": _average(
                    _coerce_float(row.get("trs_speaker")) for row in pipeline_rows
                ),
                "avg_runtime": _average(
                    _coerce_float(row.get("runtime_seconds")) for row in pipeline_rows
                ),
            }
        )
    return sorted(
        ranking,
        key=lambda row: (
            row["avg_trs_text"] is None,
            0.0 if row["avg_trs_text"] is None else -float(row["avg_trs_text"]),
            row["avg_bert_f2"] is None,
            0.0 if row["avg_bert_f2"] is None else -float(row["avg_bert_f2"]),
        ),
    )


def _best_by_sample(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    sample_ids = dict.fromkeys(str(row.get("sample_id", "")) for row in rows)
    best_rows = []
    for sample_id in sample_ids:
        sample_rows = [row for row in rows if row.get("sample_id") == sample_id]
        best_trs = _max_row(sample_rows, "trs_text")
        best_bert = _max_row(sample_rows, "bert_f2")
        first = sample_rows[0] if sample_rows else {}
        best_rows.append(
            {
                "sample_id": sample_id,
                "overlap_level": first.get("overlap_level", ""),
                "ovr": first.get("ovr"),
                "best_trs_pipeline": best_trs.get("pipeline", "") if best_trs else "",
                "best_trs_text": best_trs.get("trs_text") if best_trs else None,
                "best_bert_pipeline": best_bert.get("pipeline", "") if best_bert else "",
                "best_bert_f2": best_bert.get("bert_f2") if best_bert else None,
            }
        )
    return best_rows


def _high_overlap_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    selected = [
        row
        for row in rows
        if (_coerce_float(row.get("ovr")) or 0.0) >= 0.30
        or str(row.get("overlap_level", "")).lower() in {"heavy", "opposite"}
    ]
    return sorted(
        selected,
        key=lambda row: (
            str(row.get("sample_id", "")),
            float("inf")
            if row.get("trs_text") is None
            else -float(_coerce_float(row.get("trs_text")) or 0.0),
        ),
    )


def _failure_notes(row: dict[str, object]) -> str:
    precision = _coerce_float(row.get("bert_precision"))
    recall = _coerce_float(row.get("bert_recall"))
    notes = []
    if recall is not None and recall < 0.80:
        notes.append("missed-content-risk")
    if precision is not None and precision < 0.80:
        notes.append("hallucination-risk")
    if not notes:
        notes.append("review-best-candidate")
    return ", ".join(notes)


def _max_row(rows: list[dict[str, object]], field: str) -> dict[str, object] | None:
    valid = [row for row in rows if _coerce_float(row.get(field)) is not None]
    if not valid:
        return None
    return max(valid, key=lambda row: _coerce_float(row.get(field)) or float("-inf"))


def _average(values: Any) -> float | None:
    numeric = [value for value in values if value is not None]
    if not numeric:
        return None
    return sum(numeric) / len(numeric)


def _metric(value: float | None) -> str:
    if value is None:
        return ""
    return f"{value:.4f}"


def _seconds(value: float | None) -> str:
    if value is None:
        return ""
    return f"{value:.2f}s"


def _coerce_float(value: object) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
