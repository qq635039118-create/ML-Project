import unittest

from overlap_asr_llm.metrics import cer, speaker_block_score, strip_speaker_labels, wer


class TestMetrics(unittest.TestCase):
    def test_cer_exact_match(self):
        self.assertEqual(cer("hello", "hello"), 0)

    def test_wer_one_substitution(self):
        self.assertEqual(wer("hello world", "hello there"), 0.5)

    def test_wer_chinese_without_spaces(self):
        self.assertLess(wer("你方是要它怎么缓解空虚", "你方是药它怎么缓解空虚"), 1.0)

    def test_strip_speaker_labels(self):
        self.assertEqual(
            strip_speaker_labels("[SPEAKER_00] 你好 [SPEAKER1] 世界").strip(),
            "你好   世界",
        )

    def test_speaker_block_score_accepts_swapped_labels(self):
        reference = {
            "speaker_1": "甲方第一句 甲方第二句",
            "speaker_2": "乙方第一句 乙方第二句",
        }
        hypothesis = {
            "SPEAKER2": "乙方第一句 乙方第二句",
            "SPEAKER1": "甲方第一句 甲方第二句",
        }
        score = speaker_block_score(reference, hypothesis)
        self.assertIsNotNone(score)
        assert score is not None
        self.assertEqual(score.cer, 0)
        self.assertEqual(score.wer, 0)
        self.assertEqual(score.mapping, {"SPEAKER1": "speaker_1", "SPEAKER2": "speaker_2"})


if __name__ == "__main__":
    unittest.main()
