from pathlib import Path
import importlib.util
import json
import os
import sys
import tempfile
import unittest


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "launch_speaker_app.py"
sys.path.insert(0, str(SCRIPT_PATH.parents[1] / "src"))
SPEC = importlib.util.spec_from_file_location("launch_speaker_app", SCRIPT_PATH)
speaker_app = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(speaker_app)


class TestSpeakerApp(unittest.TestCase):
    def test_recommended_pipeline_follows_project_findings(self):
        self.assertEqual(
            speaker_app.recommended_pipeline("none", 2),
            "diarization_turn_asr",
        )
        self.assertEqual(
            speaker_app.recommended_pipeline("medium", 2),
            "diarization_asr",
        )
        self.assertEqual(
            speaker_app.recommended_pipeline("severe", 2),
            "separation_asr",
        )
        self.assertEqual(speaker_app.recommended_pipeline("heavy", 1), "direct_asr")
        self.assertEqual(
            speaker_app.recommended_pipelines("heavy", 2),
            ["diarization_asr"],
        )

    def test_subtitle_and_speaker_outputs_are_readable(self):
        segments = [
            {"start": 1.2, "end": 2.4, "speaker": "SPEAKER_2", "text": "第二句"},
            {"start": 0.0, "end": 1.0, "speaker": "SPEAKER_1", "text": "第一句"},
            {"start": 2.5, "end": 3.0, "speaker": "SPEAKER_1", "text": "第三句"},
        ]

        subtitle = speaker_app.subtitle_text(segments)
        self.assertIn("1\n00:00:00,000 --> 00:00:01,000", subtitle)
        self.assertIn("[SPEAKER_1] 第一句", subtitle)
        self.assertIn("2\n00:00:01,200 --> 00:00:02,400", subtitle)

        grouped = speaker_app.speaker_joined_text(segments)
        self.assertIn("SPEAKER_1:\n第一句 第三句", grouped)
        self.assertIn("SPEAKER_2:\n第二句", grouped)

    def test_overlap_ratio_detects_multi_speaker_overlap(self):
        turns = [
            {"start": 0.0, "end": 2.0, "speaker": "A"},
            {"start": 1.0, "end": 3.0, "speaker": "B"},
        ]
        self.assertAlmostEqual(speaker_app._turn_overlap_ratio(turns), 1 / 3)
        self.assertEqual(speaker_app._level_from_ratio(0.16), "heavy")
        self.assertEqual(speaker_app._level_from_ratio(0.4), "severe")
        self.assertEqual(speaker_app._speaker_count_from_turns(turns), 2)

    def test_auto_overlap_error_summary_runs_no_pipeline(self):
        summary = json.loads(
            speaker_app._auto_overlap_error_summary(
                "pyannote:pyannote/speaker-diarization-community-1",
                "LocalEntryNotFoundError: missing cache",
            )
        )
        self.assertIsNone(summary["selected_pipeline"])
        self.assertIsNone(summary["overlap_ratio"])
        self.assertEqual(summary["ran_pipeline_count"], 0)
        self.assertIn("HF_TOKEN", " ".join(summary["next_steps"]))

    def test_frontend_env_file_is_loaded_and_remembered(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            env_path = Path(tmpdir) / ".env"
            env_path.write_text("OVERLAP_TEST_FRONTEND_ENV=loaded\n", encoding="utf-8")
            try:
                os.environ.pop("OVERLAP_TEST_FRONTEND_ENV", None)
                os.environ.pop("OVERLAP_ASR_LLM_ENV_FILE", None)
                os.environ.pop("OVERLAP_ASR_LLM_CACHE_DIR", None)
                os.environ.pop("HF_HOME", None)
                os.environ.pop("HF_HUB_CACHE", None)
                loaded_path = speaker_app._load_frontend_env(env_path)
                self.assertEqual(loaded_path, env_path)
                self.assertEqual(os.environ["OVERLAP_TEST_FRONTEND_ENV"], "loaded")
                self.assertEqual(os.environ["OVERLAP_ASR_LLM_ENV_FILE"], str(env_path))
                self.assertEqual(
                    os.environ["OVERLAP_ASR_LLM_CACHE_DIR"],
                    str(speaker_app.DEFAULT_CACHE_DIR.resolve()),
                )
                self.assertEqual(
                    os.environ["HF_HOME"],
                    str(speaker_app.DEFAULT_CACHE_DIR.resolve() / "huggingface"),
                )
            finally:
                os.environ.pop("OVERLAP_TEST_FRONTEND_ENV", None)
                os.environ.pop("OVERLAP_ASR_LLM_ENV_FILE", None)
                os.environ.pop("OVERLAP_ASR_LLM_CACHE_DIR", None)
                os.environ.pop("HF_HOME", None)
                os.environ.pop("HF_HUB_CACHE", None)

    def test_available_model_choices_keep_mock_fallbacks(self):
        self.assertIn("mock", speaker_app._available_asr_choices())
        self.assertIn("mock", speaker_app._available_diarization_choices())
        self.assertIn("mock", speaker_app._available_separation_choices())
        self.assertIn("mock", speaker_app._available_llm_choices())
        self.assertEqual(speaker_app._default_choice(["mock"], "api"), "mock")

    def test_output_files_are_written(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            subtitle_path, speaker_path = speaker_app._write_output_files(
                Path(tmpdir),
                "1\n00:00:00,000 --> 00:00:01,000\n[SPEAKER_1] 你好",
                "SPEAKER_1:\n你好",
            )
            self.assertTrue(Path(subtitle_path).exists())
            self.assertTrue(Path(speaker_path).exists())
            self.assertEqual(Path(subtitle_path).suffix, ".srt")
            self.assertIn("SPEAKER_1", Path(speaker_path).read_text(encoding="utf-8"))

    def test_cleanup_previous_runs_keeps_current_result_and_upload(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            app_dir = Path(tmpdir) / "speaker_app"
            old_run = app_dir / "20260101_000000_old"
            current_run = app_dir / "20260101_000001_current"
            uploads = app_dir / "uploads"
            old_upload = uploads / "old_upload"
            current_upload = uploads / "current_upload"
            for path in (old_run, current_run, old_upload, current_upload):
                path.mkdir(parents=True)
                (path / "marker.txt").write_text("x", encoding="utf-8")

            deleted = speaker_app._cleanup_previous_speaker_app_runs(
                current_run,
                current_upload_dir=current_upload,
                speaker_app_dir=app_dir,
            )

            self.assertFalse(old_run.exists())
            self.assertFalse(old_upload.exists())
            self.assertTrue(current_run.exists())
            self.assertTrue(current_upload.exists())
            self.assertEqual(len(deleted), 2)

    def test_web_app_serves_index_and_options(self):
        if speaker_app.FastAPI is None:
            self.skipTest("fastapi is not installed")
        app = speaker_app.build_web_app()
        self.assertEqual(app.title, "Overlap ASR Transcript Workbench")
        self.assertIn("Overlap ASR", speaker_app.WEB_INDEX_HTML)
        self.assertIn("/api/transcribe", {route.path for route in app.routes})
        self.assertIn("domain:product_debate", [item[1] for item in speaker_app.RAG_DOMAIN_CHOICES])
        self.assertNotIn("Mock mode", speaker_app.WEB_INDEX_HTML)

    def test_llm_json_is_formatted_for_comparison(self):
        formatted = speaker_app._format_llm_output(
            json.dumps(
                {
                    "refined_text": "[SPEAKER_1] 你好。",
                    "changes": ["补全标点"],
                    "uncertain_spans": ["00:00:01 词语不确定"],
                    "hallucination_risk": "low",
                },
                ensure_ascii=False,
            )
        )
        self.assertEqual(formatted["hallucination_risk"], "low")
        self.assertIn("关键优化", formatted["display_text"])
        self.assertIn("幻觉风险: low", formatted["display_text"])


if __name__ == "__main__":
    unittest.main()
