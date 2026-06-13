"""Output helpers."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from .pipelines import PipelineResult


FIELDNAMES = [
    "sample_id",
    "audio_path",
    "overlap_level",
    "pipeline",
    "model",
    "text",
    "speaker_labels",
    "runtime_seconds",
    "cer",
    "wer",
    "text_cer",
    "text_wer",
    "error",
    "segments",
]


def write_results(
    results: list[PipelineResult],
    output_dir: Path,
    base_dir: Path | None = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = [_public_result_row(result, base_dir) for result in results]

    with (output_dir / "results.json").open("w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)

    with (output_dir / "results.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(_csv_safe_rows(rows))

    write_summary(results, output_dir / "run_summary.md")
    if any(result.pipeline == "diarization_asr" for result in results):
        write_diarization_segments(results, output_dir, base_dir)
    if any(result.pipeline == "separation_asr" for result in results):
        write_separation_segments(results, output_dir, base_dir)


def _public_path(path: str, base_dir: Path | None) -> str:
    path_obj = Path(path)
    if base_dir is None:
        return path
    try:
        return path_obj.relative_to(base_dir).as_posix()
    except ValueError:
        try:
            parent_relative = path_obj.relative_to(base_dir.parent).as_posix()
            return f"../{parent_relative}"
        except ValueError:
            return path_obj.name


def _public_result_row(
    result: PipelineResult,
    base_dir: Path | None,
) -> dict[str, object]:
    row = result.to_dict()
    row["audio_path"] = _public_path(str(row["audio_path"]), base_dir)
    row["segments"] = _public_segments(row.get("segments", []), base_dir)
    if result.pipeline == "diarization_asr":
        row["cer"] = None
        row["wer"] = None
    return row


def _public_segments(
    segments: object,
    base_dir: Path | None,
) -> list[dict[str, object]]:
    public_segments = []
    if not isinstance(segments, list):
        return public_segments
    for segment in segments:
        if not isinstance(segment, dict):
            continue
        public_segment = dict(segment)
        if "separated_audio_path" in public_segment:
            public_segment["separated_audio_path"] = _public_path(
                str(public_segment["separated_audio_path"]),
                base_dir,
            )
        public_segments.append(public_segment)
    return public_segments


def _csv_safe_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    csv_rows = []
    for row in rows:
        csv_row = dict(row)
        csv_row["segments"] = json.dumps(csv_row.get("segments", []), ensure_ascii=False)
        csv_rows.append(csv_row)
    return csv_rows


def write_diarization_segments(
    results: list[PipelineResult],
    output_dir: Path,
    base_dir: Path | None = None,
) -> None:
    rows = []
    for result in results:
        if result.pipeline != "diarization_asr" or result.error:
            continue
        for index, segment in enumerate(result.segments, start=1):
            rows.append(
                {
                    "sample_id": result.sample_id,
                    "audio_path": _public_path(result.audio_path, base_dir),
                    "overlap_level": result.overlap_level,
                    "pipeline": result.pipeline,
                    "model": result.model,
                    "segment_index": index,
                    "start": segment.get("start", ""),
                    "end": segment.get("end", ""),
                    "speaker": segment.get("speaker", "UNKNOWN"),
                    "text": segment.get("text", ""),
                }
            )

    with (output_dir / "diarization_segments.json").open("w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)

    fieldnames = [
        "sample_id",
        "audio_path",
        "overlap_level",
        "pipeline",
        "model",
        "segment_index",
        "start",
        "end",
        "speaker",
        "text",
    ]
    with (output_dir / "diarization_segments.csv").open(
        "w", encoding="utf-8", newline=""
    ) as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_separation_segments(
    results: list[PipelineResult],
    output_dir: Path,
    base_dir: Path | None = None,
) -> None:
    rows = []
    for result in results:
        if result.pipeline != "separation_asr" or result.error:
            continue
        for segment in result.segments:
            rows.append(
                {
                    "sample_id": result.sample_id,
                    "audio_path": _public_path(result.audio_path, base_dir),
                    "overlap_level": result.overlap_level,
                    "source_index": segment.get("source_index", ""),
                    "separated_audio_path": _public_path(
                        str(segment.get("separated_audio_path", "")),
                        base_dir,
                    ),
                    "start": segment.get("start", ""),
                    "end": segment.get("end", ""),
                    "speaker": segment.get("speaker", "UNKNOWN"),
                    "text": segment.get("text", ""),
                }
            )

    with (output_dir / "separation_segments.json").open("w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)

    fieldnames = [
        "sample_id",
        "audio_path",
        "overlap_level",
        "source_index",
        "separated_audio_path",
        "start",
        "end",
        "speaker",
        "text",
    ]
    with (output_dir / "separation_segments.csv").open(
        "w", encoding="utf-8", newline=""
    ) as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_summary(results: list[PipelineResult], path: Path) -> None:
    if results and all(result.pipeline == "diarization_asr" for result in results):
        write_diarization_summary(results, path)
        return
    if results and all(result.pipeline == "separation_asr" for result in results):
        write_separation_summary(results, path)
        return

    lines = [
        "# Run Summary",
        "",
        "| Sample | Overlap | Pipeline | Model | CER | WER | Runtime | Error |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | --- |",
    ]
    for result in results:
        cer_value = "" if result.cer is None else f"{result.cer:.4f}"
        wer_value = "" if result.wer is None else f"{result.wer:.4f}"
        error = result.error or ""
        lines.append(
            "| "
            f"{result.sample_id} | "
            f"{result.overlap_level} | "
            f"{result.pipeline} | "
            f"{result.model} | "
            f"{cer_value} | "
            f"{wer_value} | "
            f"{result.runtime_seconds:.4f} | "
            f"{error} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_separation_summary(results: list[PipelineResult], path: Path) -> None:
    lines = [
        "# Run Summary",
        "",
        "| Sample | Overlap | Pipeline | Model | Separated Sources | Sources With Text | CER | WER | Runtime | Error |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for result in results:
        sources = {
            str(segment.get("source_index", segment.get("speaker", "")))
            for segment in result.segments
            if segment.get("source_index", segment.get("speaker", ""))
        }
        sources_with_text = {
            str(segment.get("source_index", segment.get("speaker", "")))
            for segment in result.segments
            if segment.get("source_index", segment.get("speaker", ""))
            and str(segment.get("text", "")).strip()
        }
        cer_value = "" if result.cer is None else f"{result.cer:.4f}"
        wer_value = "" if result.wer is None else f"{result.wer:.4f}"
        lines.append(
            "| "
            f"{result.sample_id} | "
            f"{result.overlap_level} | "
            f"{result.pipeline} | "
            f"{result.model} | "
            f"{len(sources)} | "
            f"{len(sources_with_text)} | "
            f"{cer_value} | "
            f"{wer_value} | "
            f"{result.runtime_seconds:.4f} | "
            f"{result.error or ''} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_diarization_summary(results: list[PipelineResult], path: Path) -> None:
    lines = [
        "# Run Summary",
        "",
        "| Sample | Overlap | Pipeline | Model | Speakers | Segments | Segments With Text | Text CER | Text WER | Runtime | Error |",
        "| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for result in results:
        speakers = ",".join(
            dict.fromkeys(str(segment.get("speaker", "UNKNOWN")) for segment in result.segments)
        )
        segments_with_text = sum(
            1 for segment in result.segments if str(segment.get("text", "")).strip()
        )
        text_cer_value = "" if result.text_cer is None else f"{result.text_cer:.4f}"
        text_wer_value = "" if result.text_wer is None else f"{result.text_wer:.4f}"
        lines.append(
            "| "
            f"{result.sample_id} | "
            f"{result.overlap_level} | "
            f"{result.pipeline} | "
            f"{result.model} | "
            f"{speakers} | "
            f"{len(result.segments)} | "
            f"{segments_with_text} | "
            f"{text_cer_value} | "
            f"{text_wer_value} | "
            f"{result.runtime_seconds:.4f} | "
            f"{result.error or ''} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
