"""Simple text metrics used by the experiment runner."""

from __future__ import annotations

import re


_CJK_RE = re.compile(r"[\u4e00-\u9fff]")
_SPEAKER_LABEL_RE = re.compile(r"\[\s*SPEAKER[\w-]*\s*\]", re.IGNORECASE)
_TOKEN_RE = re.compile(r"[\u4e00-\u9fff]|[a-z0-9]+", re.IGNORECASE)
_PUNCTUATION_RE = re.compile(r"[^\w\s\u4e00-\u9fff]", re.UNICODE)


def strip_speaker_labels(text: str) -> str:
    return _SPEAKER_LABEL_RE.sub(" ", text)


def normalize_text(text: str) -> str:
    text = text.lower().strip()
    text = _PUNCTUATION_RE.sub("", text)
    text = re.sub(r"\s+", " ", text)
    return text


def edit_distance(left: list[str], right: list[str]) -> int:
    previous = list(range(len(right) + 1))
    for i, left_item in enumerate(left, start=1):
        current = [i]
        for j, right_item in enumerate(right, start=1):
            cost = 0 if left_item == right_item else 1
            current.append(
                min(
                    previous[j] + 1,
                    current[j - 1] + 1,
                    previous[j - 1] + cost,
                )
            )
        previous = current
    return previous[-1]


def cer(reference: str, hypothesis: str) -> float:
    ref = list(normalize_text(reference).replace(" ", ""))
    hyp = list(normalize_text(hypothesis).replace(" ", ""))
    if not ref:
        return 0.0 if not hyp else 1.0
    return edit_distance(ref, hyp) / len(ref)


def tokenize_words(text: str) -> list[str]:
    normalized = normalize_text(text)
    if _CJK_RE.search(normalized):
        try:
            import jieba  # type: ignore
        except ImportError:
            return _TOKEN_RE.findall(normalized)
        return [token for token in jieba.lcut(normalized) if token.strip()]
    return normalized.split()


def wer(reference: str, hypothesis: str) -> float:
    ref = tokenize_words(reference)
    hyp = tokenize_words(hypothesis)
    if not ref:
        return 0.0 if not hyp else 1.0
    return edit_distance(ref, hyp) / len(ref)
