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
    "error",
]


def write_results(results: list[PipelineResult], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = [result.to_dict() for result in results]

    with (output_dir / "results.json").open("w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)

    with (output_dir / "results.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    write_summary(results, output_dir / "run_summary.md")
    write_qualitative_review_template(
        results, output_dir / "qualitative_review_template.csv"
    )


def write_summary(results: list[PipelineResult], path: Path) -> None:
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


def write_qualitative_review_template(
    results: list[PipelineResult],
    path: Path,
) -> None:
    """Write a human review sheet for readability and failure-case analysis."""
    fieldnames = [
        "sample_id",
        "overlap_level",
        "pipeline",
        "manual_readability_1_to_5",
        "failure_type",
        "observation",
        "hallucination_risk",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow(
                {
                    "sample_id": result.sample_id,
                    "overlap_level": result.overlap_level,
                    "pipeline": result.pipeline,
                    "manual_readability_1_to_5": "",
                    "failure_type": "",
                    "observation": "",
                    "hallucination_risk": "",
                }
            )
