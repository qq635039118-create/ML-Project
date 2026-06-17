"""Command-line entry point for the experiment suite."""

from __future__ import annotations

import argparse
from dataclasses import replace
import os
from pathlib import Path

from .config import load_config
from .io import write_results
from .pipelines import run_all
from .readability import DEFAULT_BERT_MODEL, evaluate_results


def _load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run overlapping speech experiments.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run all experiment pipelines.")
    run_parser.add_argument(
        "--config",
        default="configs/mock.json",
        help="Path to the JSON experiment config.",
    )
    run_parser.add_argument(
        "--mock",
        action="store_true",
        help="Force all providers to mock mode for local smoke tests.",
    )
    run_parser.add_argument(
        "--incremental",
        action="store_true",
        help="Write outputs after each sample so long experiments keep partial results.",
    )

    evaluate_parser = subparsers.add_parser(
        "evaluate",
        help="Compute post-run readability metrics for an existing results.json.",
    )
    evaluate_parser.add_argument(
        "--config",
        default="configs/all_pipelines.json",
        help="Path to the JSON experiment config used for the run.",
    )
    evaluate_parser.add_argument(
        "--results",
        required=True,
        help="Path to an existing results.json file.",
    )
    evaluate_parser.add_argument(
        "--device",
        choices=["auto", "cuda", "cpu"],
        default="auto",
        help="Device for BERTScore evaluation.",
    )
    evaluate_parser.add_argument(
        "--bert-model",
        default=DEFAULT_BERT_MODEL,
        help="Model name passed to BERTScore.",
    )
    evaluate_parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size for BERTScore evaluation.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "run":
        config_path = Path(args.config)
        _load_env_file(config_path.resolve().parent.parent / ".env")
        config = load_config(Path(args.config))
        if args.mock:
            config.models.update(
                {"asr": "mock", "diarization": "mock", "separation": "mock", "llm": "mock"}
            )
        if args.incremental:
            results = []
            for sample in config.samples:
                sample_config = replace(config, samples=[sample])
                results.extend(run_all(sample_config))
                write_results(results, config.output_dir, config.base_dir)
        else:
            results = run_all(config)
        write_results(results, config.output_dir, config.base_dir)
        print(f"Wrote results to {config.output_dir}")
        return 0

    if args.command == "evaluate":
        config_path = Path(args.config)
        _load_env_file(config_path.resolve().parent.parent / ".env")
        config = load_config(config_path)
        evaluation = evaluate_results(
            config=config,
            results_path=Path(args.results),
            device=args.device,
            bert_model=args.bert_model,
            batch_size=args.batch_size,
        )
        print(f"Wrote readability evaluation to {evaluation.output_dir}")
        return 0

    parser.error(f"Unsupported command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
