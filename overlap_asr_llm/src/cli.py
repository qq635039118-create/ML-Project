"""Command-line entry point for the experiment suite."""

from __future__ import annotations

import argparse
from pathlib import Path

from .config import load_config
from .io import write_results
from .pipelines import run_all


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run overlapping speech experiments.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run all experiment pipelines.")
    run_parser.add_argument(
        "--config",
        default="configs/experiment.json",
        help="Path to the JSON experiment config.",
    )
    run_parser.add_argument(
        "--mock",
        action="store_true",
        help="Force all providers to mock mode for local smoke tests.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "run":
        config = load_config(Path(args.config))
        if args.mock:
            config.models.update(
                {"asr": "mock", "diarization": "mock", "separation": "mock", "llm": "mock"}
            )
        results = run_all(config)
        write_results(results, config.output_dir)
        print(f"Wrote results to {config.output_dir}")
        return 0

    parser.error(f"Unsupported command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
