"""Stdlib-only smoke test for the project runner."""

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from overlap_asr_llm.config import load_config
from overlap_asr_llm.io import write_results
from overlap_asr_llm.pipelines import run_all


def main() -> int:
    config = load_config(ROOT / "configs" / "experiment.json")
    config.models.update(
        {"asr": "mock", "diarization": "mock", "separation": "mock", "llm": "mock"}
    )
    config.pipelines[:] = [
        "direct_asr",
        "diarization_asr",
        "separation_asr",
        "llm_rag_refine",
    ]
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
