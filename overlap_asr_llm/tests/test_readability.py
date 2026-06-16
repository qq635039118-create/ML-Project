import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from overlap_asr_llm.cli import main
from overlap_asr_llm.config import ExperimentConfig, Sample
from overlap_asr_llm.readability import (
    bert_f_beta,
    compute_overlap_ratio,
    evaluate_results,
    trs_speaker,
    trs_text,
)


class TestReadability(unittest.TestCase):
    def test_compute_overlap_ratio_returns_none_without_segments(self):
        self.assertIsNone(compute_overlap_ratio([]))

    def test_compute_overlap_ratio_partial_overlap(self):
        segments = [
            {"speaker": "A", "start": 0, "end": 10},
            {"speaker": "B", "start": 5, "end": 15},
        ]
        self.assertAlmostEqual(compute_overlap_ratio(segments), 5 / 15)

    def test_compute_overlap_ratio_full_overlap(self):
        segments = [
            {"speaker": "A", "start": 0, "end": 10},
            {"speaker": "B", "start": 0, "end": 10},
        ]
        self.assertEqual(compute_overlap_ratio(segments), 1.0)

    def test_compute_overlap_ratio_multiple_segments(self):
        segments = [
            {"speaker": "A", "start": 0, "end": 4},
            {"speaker": "B", "start": 2, "end": 6},
            {"speaker": "A", "start": 8, "end": 10},
        ]
        self.assertAlmostEqual(compute_overlap_ratio(segments), 2 / 8)

    def test_bert_f_beta_weights_recall(self):
        high_recall = bert_f_beta(0.5, 1.0, beta=2.0)
        high_precision = bert_f_beta(1.0, 0.5, beta=2.0)
        self.assertIsNotNone(high_recall)
        self.assertIsNotNone(high_precision)
        assert high_recall is not None
        assert high_precision is not None
        self.assertGreater(high_recall, high_precision)

    def test_trs_text_uses_plain_geometric_mean(self):
        score = trs_text(0.25, 0.81)
        self.assertAlmostEqual(score or 0.0, 100 * math.sqrt(0.75 * 0.81))
        self.assertIsNone(trs_text(None, 0.81))
        self.assertIsNone(trs_text(0.25, None))

    def test_trs_speaker_requires_speaker_metric(self):
        self.assertIsNone(trs_speaker(0.1, 0.9, None))
        score = trs_speaker(0.1, 0.9, 0.2)
        self.assertAlmostEqual(score or 0.0, 100 * ((0.9 * 0.9 * 0.8) ** (1 / 3)))

    def test_evaluate_results_writes_outputs_with_mock_scorer(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            config = ExperimentConfig(
                project_name="test",
                output_dir=tmp_path,
                language="zh",
                asr_prompt=None,
                models={},
                asr_models=[],
                pipelines=[],
                llm_rag_sources=(),
                rag_context=[],
                samples=[
                    Sample(
                        id="sample1",
                        audio_path=tmp_path / "sample.wav",
                        overlap_level="medium",
                        speakers=2,
                        reference="你好 世界",
                    )
                ],
                base_dir=tmp_path,
            )
            results_path = tmp_path / "results.json"
            results_path.write_text(
                json.dumps(
                    [
                        {
                            "sample_id": "sample1",
                            "overlap_level": "medium",
                            "pipeline": "direct_asr",
                            "model": "mock",
                            "text": "你好 世界",
                            "runtime_seconds": 1.0,
                            "cer": 0.0,
                            "wer": 0.0,
                            "speaker_block_cer": None,
                            "error": None,
                            "segments": [],
                        },
                        {
                            "sample_id": "sample1",
                            "overlap_level": "medium",
                            "pipeline": "diarization_asr",
                            "model": "mock",
                            "text": "",
                            "runtime_seconds": 2.0,
                            "cer": 0.1,
                            "wer": 0.2,
                            "speaker_block_cer": 0.25,
                            "error": None,
                            "segments": [
                                {"speaker": "A", "start": 0, "end": 10, "text": "你好"},
                                {"speaker": "B", "start": 5, "end": 15, "text": "世界"},
                            ],
                        },
                    ],
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            def scorer(hypotheses, references, *, model_type, device, batch_size):
                self.assertEqual(model_type, "mock-bert")
                self.assertEqual(device, "cpu")
                self.assertEqual(batch_size, 2)
                return [(0.9, 0.8, 0.85) for _ in hypotheses]

            evaluation = evaluate_results(
                config,
                results_path,
                device="cpu",
                bert_model="mock-bert",
                batch_size=2,
                scorer=scorer,
            )

            self.assertEqual(len(evaluation.rows), 2)
            self.assertTrue((tmp_path / "readability_results.json").exists())
            self.assertTrue((tmp_path / "readability_results.csv").exists())
            self.assertTrue((tmp_path / "readability_summary.md").exists())
            self.assertAlmostEqual(evaluation.rows[0]["ovr"], 5 / 15)
            self.assertEqual(evaluation.rows[0]["ovr_source"], "estimated")
            self.assertIsNotNone(evaluation.rows[0]["trs_text"])
            self.assertIsNone(evaluation.rows[0]["trs_speaker"])
            self.assertIsNotNone(evaluation.rows[1]["trs_speaker"])

    def test_evaluate_cli_passes_arguments(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            config_path = tmp_path / "config.json"
            results_path = tmp_path / "results.json"
            config_path.write_text(
                json.dumps(
                    {
                        "project_name": "test",
                        "output_dir": "outputs",
                        "language": "zh",
                        "models": {},
                        "samples": [
                            {
                                "id": "sample1",
                                "audio_path": "sample.wav",
                                "overlap_level": "none",
                                "speakers": 1,
                                "reference": "你好",
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            results_path.write_text("[]", encoding="utf-8")

            with patch("overlap_asr_llm.cli.evaluate_results") as evaluate:
                evaluate.return_value.output_dir = tmp_path
                exit_code = main(
                    [
                        "evaluate",
                        "--config",
                        str(config_path),
                        "--results",
                        str(results_path),
                        "--device",
                        "cpu",
                        "--bert-model",
                        "mock-bert",
                        "--batch-size",
                        "3",
                    ]
                )

            self.assertEqual(exit_code, 0)
            kwargs = evaluate.call_args.kwargs
            self.assertEqual(kwargs["results_path"], results_path)
            self.assertEqual(kwargs["device"], "cpu")
            self.assertEqual(kwargs["bert_model"], "mock-bert")
            self.assertEqual(kwargs["batch_size"], 3)


if __name__ == "__main__":
    unittest.main()
