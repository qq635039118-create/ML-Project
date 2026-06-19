"""Lightweight tag-based RAG helpers for transcript refinement."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class KnowledgeCard:
    id: str
    tags: tuple[str, ...]
    content: str


DEFAULT_KNOWLEDGE_CARDS = (
    KnowledgeCard(
        id="task_basic",
        tags=("task:speaker_transcript", "domain:debate"),
        content=(
            "The audio is a two-speaker Mandarin debate. Preserve the original "
            "meaning, speaker labels, and timestamp order."
        ),
    ),
    KnowledgeCard(
        id="no_hallucination",
        tags=("safety:faithfulness",),
        content=(
            "Do not add words, claims, examples, or arguments that are not "
            "supported by the transcript."
        ),
    ),
    KnowledgeCard(
        id="speaker_labels",
        tags=("pipeline:diarization_asr", "pipeline:speaker_transcript"),
        content=(
            "Keep existing speaker labels. Do not swap speakers or invent speaker "
            "identities."
        ),
    ),
    KnowledgeCard(
        id="heavy_overlap",
        tags=("overlap:heavy", "overlap:opposite"),
        content=(
            "Severe overlap may cause missing words. Mark uncertain spans instead "
            "of guessing the missing content."
        ),
    ),
    KnowledgeCard(
        id="domain_terms",
        tags=("domain:product_debate",),
        content=(
            "Important terms may include \u4ea7\u54c1\u67b6\u6784\u5347\u7ea7, "
            "\u81ea\u52a8\u5316\u6548\u7387, \u7528\u6237\u4f53\u9a8c, "
            "\u56e2\u961f\u51b3\u7b56, \u65e7\u63a5\u53e3, \u4e0a\u7ebf, "
            "\u6570\u636e\u4e0b\u6ed1."
        ),
    ),
    KnowledgeCard(
        id="education_terms",
        tags=("domain:education",),
        content=(
            "Education transcripts may contain terms such as \u4e2a\u6027\u5316\u5b66\u4e60, "
            "\u6559\u5b66\u76ee\u6807, \u8bfe\u5802\u4e92\u52a8, \u4f5c\u4e1a\u6279\u6539, "
            "\u5b66\u4e60\u53cd\u9988, \u6559\u5e08\u8bc4\u4ef7, \u8bfe\u7a0b\u8bbe\u8ba1."
        ),
    ),
    KnowledgeCard(
        id="healthcare_terms",
        tags=("domain:healthcare",),
        content=(
            "Healthcare transcripts may contain terms such as \u75c5\u53f2, \u8bca\u65ad, "
            "\u75c7\u72b6, \u7528\u836f, \u5242\u91cf, \u590d\u8bca, "
            "\u68c0\u67e5\u7ed3\u679c. Do not turn uncertain medical wording into advice."
        ),
    ),
    KnowledgeCard(
        id="finance_terms",
        tags=("domain:finance",),
        content=(
            "Finance transcripts may contain terms such as \u6536\u5165, \u6210\u672c, "
            "\u5229\u6da6\u7387, \u73b0\u91d1\u6d41, \u9884\u7b97, \u98ce\u9669\u655e\u53e3, "
            "\u6295\u8d44\u56de\u62a5\u7387, \u540c\u6bd4, \u73af\u6bd4."
        ),
    ),
    KnowledgeCard(
        id="legal_terms",
        tags=("domain:legal",),
        content=(
            "Legal transcripts may contain terms such as \u5408\u540c, \u6761\u6b3e, "
            "\u8bc1\u636e, \u8d23\u4efb, \u8fdd\u7ea6, \u5408\u89c4, \u6388\u6743, "
            "\u4fdd\u5bc6\u534f\u8bae. Preserve quoted wording carefully."
        ),
    ),
    KnowledgeCard(
        id="technology_terms",
        tags=("domain:technology",),
        content=(
            "Technology transcripts may contain terms such as API, \u6570\u636e\u5e93, "
            "\u67b6\u6784, \u90e8\u7f72, \u63a5\u53e3, \u6a21\u578b, \u5ef6\u8fdf, "
            "\u541e\u5410\u91cf, \u76d1\u63a7, \u56de\u6eda."
        ),
    ),
    KnowledgeCard(
        id="meeting_terms",
        tags=("domain:meeting",),
        content=(
            "Meeting transcripts may contain agenda items, owners, deadlines, "
            "blockers, decisions, and action items. Keep them separate from the "
            "spoken transcript unless explicitly present."
        ),
    ),
    KnowledgeCard(
        id="customer_service_terms",
        tags=("domain:customer_service",),
        content=(
            "Customer service transcripts may contain terms such as \u5de5\u5355, \u9000\u6b3e, "
            "\u8ba2\u5355\u53f7, \u6295\u8bc9, \u552e\u540e, \u5347\u7ea7\u5904\u7406, "
            "\u6ee1\u610f\u5ea6, \u670d\u52a1\u534f\u8bae."
        ),
    ),
    KnowledgeCard(
        id="research_terms",
        tags=("domain:research",),
        content=(
            "Research transcripts may contain terms such as \u5b9e\u9a8c\u8bbe\u8ba1, "
            "\u57fa\u7ebf\u6a21\u578b, \u6d88\u878d\u5b9e\u9a8c, \u6570\u636e\u96c6, "
            "\u6307\u6807, \u663e\u8457\u6027, \u8bef\u5dee\u5206\u6790, \u590d\u73b0."
        ),
    ),
)


DOMAIN_KEYWORDS = {
    "domain:education": ("\u6559\u80b2", "\u5b66\u751f", "\u8001\u5e08", "\u6559\u5e08", "\u8bfe\u5802", "\u8bfe\u7a0b", "\u4f5c\u4e1a", "\u5b66\u4e60"),
    "domain:healthcare": ("\u533b\u7597", "\u533b\u751f", "\u60a3\u8005", "\u75c5\u4eba", "\u8bca\u65ad", "\u75c7\u72b6", "\u7528\u836f", "\u590d\u8bca"),
    "domain:finance": ("\u91d1\u878d", "\u8d22\u52a1", "\u9884\u7b97", "\u6536\u5165", "\u6210\u672c", "\u5229\u6da6", "\u73b0\u91d1\u6d41", "\u6295\u8d44"),
    "domain:legal": ("\u6cd5\u5f8b", "\u5408\u540c", "\u6761\u6b3e", "\u8bc1\u636e", "\u8d23\u4efb", "\u8fdd\u7ea6", "\u5408\u89c4", "\u6388\u6743"),
    "domain:technology": ("\u6280\u672f", "\u67b6\u6784", "\u63a5\u53e3", "API", "\u6570\u636e\u5e93", "\u90e8\u7f72", "\u6a21\u578b", "\u7b97\u6cd5"),
    "domain:meeting": ("\u4f1a\u8bae", "\u8bae\u7a0b", "\u7eaa\u8981", "\u8d1f\u8d23\u4eba", "\u622a\u6b62", "\u51b3\u7b56", "\u884c\u52a8\u9879", "\u6392\u671f"),
    "domain:customer_service": ("\u5ba2\u670d", "\u5ba2\u6237", "\u5de5\u5355", "\u9000\u6b3e", "\u8ba2\u5355", "\u6295\u8bc9", "\u552e\u540e", "\u6ee1\u610f\u5ea6"),
    "domain:research": ("\u7814\u7a76", "\u8bba\u6587", "\u5b9e\u9a8c", "\u6570\u636e\u96c6", "\u6307\u6807", "\u6a21\u578b", "\u57fa\u7ebf", "\u590d\u73b0"),
}


def tags_for_sample(overlap_level: str, pipeline: str) -> list[str]:
    tags = [
        "task:speaker_transcript",
        "domain:debate",
        "domain:product_debate",
        "safety:faithfulness",
        f"pipeline:{pipeline}",
    ]
    if overlap_level:
        tags.append(f"overlap:{overlap_level}")
    return tags


def infer_domain_tags(text: str) -> list[str]:
    tags: list[str] = []
    for tag, keywords in DOMAIN_KEYWORDS.items():
        if any(keyword in text for keyword in keywords):
            tags.append(tag)
    return tags


def retrieve_rag_context(
    tags: list[str],
    base_context: list[str] | None = None,
    limit: int = 8,
) -> list[str]:
    selected: list[str] = []
    required_ids = {"task_basic", "no_hallucination"}
    tag_set = set(tags)

    for card in DEFAULT_KNOWLEDGE_CARDS:
        if card.id in required_ids or tag_set.intersection(card.tags):
            selected.append(card.content)
        if len(selected) >= limit:
            break

    for item in base_context or []:
        if item and item not in selected:
            selected.append(item)
        if len(selected) >= limit:
            break

    return selected
