from pathlib import Path
import tempfile
import unittest

from overlap_asr_llm.config import load_config
from overlap_asr_llm.io import write_results
from overlap_asr_llm.pipelines import run_all


class TestRunner(unittest.TestCase):
    def test_mock_runner_produces_four_results_per_sample(self):
        config = load_config(Path("configs/experiment.json"))
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

    def test_writer_produces_core_result_files_only(self):
        config = load_config(Path("configs/experiment.json"))
        config.models.update(
            {"asr": "mock", "diarization": "mock", "separation": "mock", "llm": "mock"}
        )
        results = run_all(config)
        with tempfile.TemporaryDirectory() as tmpdir:
            write_results(results, Path(tmpdir))
            self.assertTrue((Path(tmpdir) / "results.csv").exists())
            self.assertTrue((Path(tmpdir) / "results.json").exists())
            self.assertTrue((Path(tmpdir) / "run_summary.md").exists())
            self.assertFalse((Path(tmpdir) / "qualitative_review_template.csv").exists())

    def test_runner_can_select_direct_asr_only(self):
        config = load_config(Path("configs/experiment.json"))
        config.models.update(
            {"asr": "mock", "diarization": "mock", "separation": "mock", "llm": "mock"}
        )
        config.pipelines[:] = ["direct_asr"]
        results = run_all(config)
        self.assertEqual(len(results), len(config.samples))
        self.assertEqual({result.pipeline for result in results}, {"direct_asr"})


if __name__ == "__main__":
    unittest.main()
