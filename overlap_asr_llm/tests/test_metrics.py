import unittest

from overlap_asr_llm.metrics import cer, wer


class TestMetrics(unittest.TestCase):
    def test_cer_exact_match(self):
        self.assertEqual(cer("hello", "hello"), 0)

    def test_wer_one_substitution(self):
        self.assertEqual(wer("hello world", "hello there"), 0.5)

    def test_wer_chinese_without_spaces(self):
        self.assertLess(wer("你方是要它怎么缓解空虚", "你方是药它怎么缓解空虚"), 1.0)


if __name__ == "__main__":
    unittest.main()
