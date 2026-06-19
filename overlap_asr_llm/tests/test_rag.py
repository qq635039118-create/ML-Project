from overlap_asr_llm.rag import infer_domain_tags, retrieve_rag_context, tags_for_sample


def test_infer_domain_tags_from_multidomain_text():
    text = (
        "\u8001\u5e08\u5728\u8bfe\u5802\u4e0a\u8ba8\u8bba\u5b66\u751f\u5b66\u4e60\u53cd\u9988,"
        "\u8d22\u52a1\u56e2\u961f\u540c\u65f6\u5173\u5fc3\u9884\u7b97\u548c\u73b0\u91d1\u6d41."
    )

    tags = infer_domain_tags(text)

    assert "domain:education" in tags
    assert "domain:finance" in tags


def test_retrieve_rag_context_selects_matching_domain_cards():
    tags = tags_for_sample("heavy", "llm_rag_refine")
    tags.extend(
        infer_domain_tags(
            "\u533b\u751f\u9700\u8981\u6839\u636e\u75c7\u72b6\u548c\u68c0\u67e5\u7ed3\u679c\u5224\u65ad\u662f\u5426\u590d\u8bca."
        )
    )

    context = retrieve_rag_context(tags, base_context=["project-specific context"], limit=8)
    joined = "\n".join(context)

    assert "Do not add words" in joined
    assert "Healthcare transcripts" in joined
    assert "Education transcripts" not in joined
    assert "project-specific context" in context
