"""Simple text metrics used by the experiment runner."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations
import re


_CJK_RE = re.compile(r"[\u4e00-\u9fff]")
_SPEAKER_LABEL_RE = re.compile(r"\[\s*SPEAKER[\w-]*\s*\]", re.IGNORECASE)
_TOKEN_RE = re.compile(r"[\u4e00-\u9fff]|[a-z0-9]+", re.IGNORECASE)
_PUNCTUATION_RE = re.compile(r"[^\w\s\u4e00-\u9fff]", re.UNICODE)


@dataclass(frozen=True)
class SpeakerBlockScore:
    cer: float
    wer: float
    mapping: dict[str, str]
    reference_text: str
    hypothesis_text: str


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


def speaker_block_score(
    reference_speakers: dict[str, str],
    hypothesis_speakers: dict[str, str],
) -> SpeakerBlockScore | None:
    """Score speaker-attributed text with the best speaker-label permutation.

    The reference is ordered by speaker blocks, for example all speaker A text
    followed by all speaker B text. Predicted labels may be swapped, so this
    evaluates all small label assignments and returns the lowest CER/WER pair.
    """

    ref_items = [
        (speaker, text.strip())
        for speaker, text in reference_speakers.items()
        if text and text.strip()
    ]
    hyp_items = [
        (speaker, text.strip())
        for speaker, text in hypothesis_speakers.items()
        if text and text.strip()
    ]
    if not ref_items:
        return None

    ref_text = " ".join(text for _, text in ref_items).strip()
    hyp_labels = [speaker for speaker, _ in hyp_items]
    hyp_text_by_label = dict(hyp_items)

    if not hyp_labels:
        return SpeakerBlockScore(
            cer=cer(ref_text, ""),
            wer=wer(ref_text, ""),
            mapping={},
            reference_text=ref_text,
            hypothesis_text="",
        )

    ref_count = len(ref_items)
    if len(hyp_labels) >= ref_count:
        assignments = permutations(hyp_labels, ref_count)
    else:
        assignments = permutations(
            hyp_labels + [None] * (ref_count - len(hyp_labels)),
            ref_count,
        )

    best: SpeakerBlockScore | None = None
    seen: set[tuple[str | None, ...]] = set()
    for assignment in assignments:
        if assignment in seen:
            continue
        seen.add(assignment)
        assigned_labels = {label for label in assignment if label is not None}
        ordered_hyp_parts = [
            hyp_text_by_label.get(label, "") if label is not None else ""
            for label in assignment
        ]
        ordered_hyp_parts.extend(
            hyp_text_by_label[label]
            for label in hyp_labels
            if label not in assigned_labels
        )
        hyp_text = " ".join(part for part in ordered_hyp_parts if part).strip()
        mapping = {
            str(label): ref_items[index][0]
            for index, label in enumerate(assignment)
            if label is not None
        }
        candidate = SpeakerBlockScore(
            cer=cer(ref_text, hyp_text),
            wer=wer(ref_text, hyp_text),
            mapping=mapping,
            reference_text=ref_text,
            hypothesis_text=hyp_text,
        )
        if best is None or (candidate.cer, candidate.wer) < (best.cer, best.wer):
            best = candidate

    return best
