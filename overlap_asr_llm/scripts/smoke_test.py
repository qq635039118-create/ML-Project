"""Stdlib-only smoke test for the project runner."""

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from src.config import load_config
from src.io import write_results
from src.pipelines import run_all


def main() -> int:
    config = load_config(ROOT / "configs" / "experiment.json")
    config.models.update(
        {"asr": "mock", "diarization": "mock", "separation": "mock", "llm": "mock"}
    )
    results = run_all(config)
    expected = len(config.samples) * 4
    if len(results) != expected:
        print(f"Expected {expected} results, got {len(results)}")
        return 1
    write_results(results, config.output_dir)
    print(f"Smoke test passed. Wrote outputs to {config.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
