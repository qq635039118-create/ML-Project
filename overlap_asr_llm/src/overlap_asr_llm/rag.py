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
            "Important terms may include 产品架构升级, 自动化效率, 用户体验, "
            "团队决策, 旧接口, 上线, 数据下滑."
        ),
    ),
)


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

