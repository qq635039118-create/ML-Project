# Run Summary

## Pipeline Ranking

| Rank | Pipeline | Runs | Avg CER | Avg WER | Avg Runtime | Wins | Errors |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | llm_rag_refine (speechbrain_diarization) | 5 | 0.2084 | 0.1968 | 78.46s | 2 | 0 |
| 2 | llm_rag_refine (pyannote_diarization) | 5 | 0.2089 | 0.1968 | 73.24s | 3 | 0 |

## Best By Overlap

| Sample | Overlap | Best Pipeline | CER | WER | Runtime | Runner-Up | Delta CER |
| --- | --- | --- | ---: | ---: | ---: | --- | ---: |
| sample2_no_overlap | none | llm_rag_refine (speechbrain_diarization) | 0.0117 | 0.0162 | 55.34s | llm_rag_refine (pyannote_diarization) | 0.0000 |
| sample2_light_overlap | light | llm_rag_refine (pyannote_diarization) | 0.0468 | 0.0486 | 72.57s | llm_rag_refine (speechbrain_diarization) | 0.0000 |
| sample2_mid_overlap | medium | llm_rag_refine (pyannote_diarization) | 0.1218 | 0.1336 | 76.92s | llm_rag_refine (speechbrain_diarization) | 0.0984 |
| sample2_heavy_overlap | heavy | llm_rag_refine (speechbrain_diarization) | 0.1522 | 0.1538 | 80.79s | llm_rag_refine (pyannote_diarization) | 0.1007 |
| sample2_opposite_overlap | opposite | llm_rag_refine (pyannote_diarization) | 0.6112 | 0.5992 | 68.41s | llm_rag_refine (speechbrain_diarization) | 0.0000 |

## Detailed Results

| Sample | Overlap | Pipeline | CER | WER | Runtime | Basis | Error |
| --- | --- | --- | ---: | ---: | ---: | --- | --- |
| sample2_no_overlap | none | llm_rag_refine (speechbrain_diarization) | 0.0117 | 0.0162 | 55.34s | timeline |  |
| sample2_no_overlap | none | llm_rag_refine (pyannote_diarization) | 0.0117 | 0.0162 | 72.54s | timeline |  |
| sample2_light_overlap | light | llm_rag_refine (speechbrain_diarization) | 0.0468 | 0.0486 | 83.79s | timeline |  |
| sample2_light_overlap | light | llm_rag_refine (pyannote_diarization) | 0.0468 | 0.0486 | 72.57s | timeline |  |
| sample2_mid_overlap | medium | llm_rag_refine (speechbrain_diarization) | 0.2201 | 0.1660 | 96.83s | timeline |  |
| sample2_mid_overlap | medium | llm_rag_refine (pyannote_diarization) | 0.1218 | 0.1336 | 76.92s | timeline |  |
| sample2_heavy_overlap | heavy | llm_rag_refine (speechbrain_diarization) | 0.1522 | 0.1538 | 80.79s | timeline |  |
| sample2_heavy_overlap | heavy | llm_rag_refine (pyannote_diarization) | 0.2529 | 0.1862 | 75.77s | timeline |  |
| sample2_opposite_overlap | opposite | llm_rag_refine (speechbrain_diarization) | 0.6112 | 0.5992 | 75.57s | timeline |  |
| sample2_opposite_overlap | opposite | llm_rag_refine (pyannote_diarization) | 0.6112 | 0.5992 | 68.41s | timeline |  |
