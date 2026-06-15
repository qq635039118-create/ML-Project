"""Command-line entry point for the experiment suite."""

from __future__ import annotations

import argparse
from dataclasses import replace
import os
from pathlib import Path

from .config import load_config
from .io import write_results
from .pipelines import run_all


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

    parser.error(f"Unsupported command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
