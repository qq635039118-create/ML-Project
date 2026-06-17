# Readability Evaluation Summary

## Pipeline Ranking

| Rank | Pipeline | Runs | Avg CER | Avg WER | Avg BERT F2 | Avg TRS Text | Avg TRS Speaker | Avg Runtime |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | diarization_asr | 5 | 0.1532 | 0.1563 | 0.8732 | 85.8221 | 85.4050 | 10.74s |
| 2 | direct_asr | 5 | 0.1888 | 0.1903 | 0.8732 | 83.4772 |  | 6.72s |
| 3 | diarization_turn_asr | 5 | 0.2276 | 0.2170 | 0.8687 | 81.7304 | 80.1656 | 25.00s |
| 4 | llm_rag_refine | 5 | 0.3372 | 0.2656 | 0.8728 | 74.3083 |  | 81.18s |
| 5 | separation_asr | 5 | 0.4595 | 0.4672 | 0.7924 | 64.1415 | 60.3207 | 29.70s |

## Best By OVR

| Sample | Overlap | OVR | Best By TRS Text | TRS Text | Best By BERT F2 | BERT F2 |
| --- | --- | ---: | --- | ---: | --- | ---: |
| sample2_no_overlap | none | 0.0000 | diarization_turn_asr | 97.9449 | diarization_turn_asr | 0.9684 |
| sample2_light_overlap | light | 0.0304 | llm_rag_refine | 95.2357 | llm_rag_refine | 0.9492 |
| sample2_mid_overlap | medium | 0.0959 | llm_rag_refine | 89.2127 | llm_rag_refine | 0.9038 |
| sample2_heavy_overlap | heavy | 0.1572 | llm_rag_refine | 86.5547 | llm_rag_refine | 0.8837 |
| sample2_opposite_overlap | opposite | 0.4680 | separation_asr | 89.0998 | separation_asr | 0.8475 |

## High Overlap Review

| Sample | OVR | Pipeline | BERT Precision | BERT Recall | BERT F2 | TRS Text | Notes |
| --- | ---: | --- | ---: | ---: | ---: | ---: | --- |
| sample2_heavy_overlap | 0.1572 | llm_rag_refine | 0.9165 | 0.8759 | 0.8837 | 86.5547 | review-best-candidate |
| sample2_heavy_overlap | 0.1572 | direct_asr | 0.8937 | 0.8467 | 0.8557 | 85.1740 | review-best-candidate |
| sample2_heavy_overlap | 0.1572 | diarization_asr | 0.8937 | 0.8467 | 0.8557 | 85.1740 | review-best-candidate |
| sample2_heavy_overlap | 0.1572 | diarization_turn_asr | 0.8240 | 0.8633 | 0.8552 | 77.7699 | review-best-candidate |
| sample2_heavy_overlap | 0.1572 | separation_asr | 0.7491 | 0.7897 | 0.7812 | 46.6596 | missed-content-risk, hallucination-risk |
| sample2_opposite_overlap | 0.4680 | separation_asr | 0.8509 | 0.8466 | 0.8475 | 89.0998 | review-best-candidate |
| sample2_opposite_overlap | 0.4680 | diarization_asr | 0.8805 | 0.7813 | 0.7993 | 68.2709 | missed-content-risk |
| sample2_opposite_overlap | 0.4680 | diarization_turn_asr | 0.6843 | 0.7298 | 0.7202 | 59.5159 | missed-content-risk, hallucination-risk |
| sample2_opposite_overlap | 0.4680 | llm_rag_refine | 0.8939 | 0.7961 | 0.8139 | 56.2516 | missed-content-risk |
| sample2_opposite_overlap | 0.4680 | direct_asr | 0.8805 | 0.7813 | 0.7993 | 55.7430 | missed-content-risk |
