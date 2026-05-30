from pathlib import Path
import unittest

from overlap_asr_llm.config import load_config
from overlap_asr_llm.pipelines import run_all


class TestRunner(unittest.TestCase):
    def test_mock_runner_produces_four_results_per_sample(self):
        config = load_config(Path("configs/experiment.json"))
        config.models.update(
            {"asr": "mock", "diarization": "mock", "separation": "mock", "llm": "mock"}
        )
        results = run_all(config)
        self.assertEqual(len(results), len(config.samples) * 4)
        self.assertEqual(
            {result.pipeline for result in results},
            {
                "direct_asr",
                "diarization_asr",
                "separation_asr",
                "llm_rag_refine",
            },
        )


if __name__ == "__main__":
    unittest.main()
