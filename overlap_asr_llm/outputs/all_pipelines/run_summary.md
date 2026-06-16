# Run Summary

## Pipeline Ranking

| Rank | Pipeline | Runs | Avg CER | Avg WER | Avg Runtime | Wins | Errors |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | diarization_asr | 5 | 0.1532 | 0.1563 | 10.74s | 0 | 0 |
| 2 | direct_asr | 5 | 0.1888 | 0.1903 | 6.72s | 1 | 0 |
| 3 | diarization_turn_asr | 5 | 0.2276 | 0.2170 | 25.00s | 1 | 0 |
| 4 | llm_rag_refine | 5 | 0.3372 | 0.2656 | 81.18s | 2 | 0 |
| 5 | separation_asr | 5 | 0.4595 | 0.4672 | 29.70s | 1 | 0 |

## Best By Overlap

| Sample | Overlap | Best Pipeline | CER | WER | Runtime | Runner-Up | Delta CER |
| --- | --- | --- | ---: | ---: | ---: | --- | ---: |
| sample2_no_overlap | none | diarization_turn_asr | 0.0094 | 0.0121 | 22.43s | direct_asr | 0.0023 |
| sample2_light_overlap | light | llm_rag_refine | 0.0445 | 0.0445 | 86.28s | direct_asr | 0.0023 |
| sample2_mid_overlap | medium | llm_rag_refine | 0.1194 | 0.1296 | 85.13s | direct_asr | 0.0023 |
| sample2_heavy_overlap | heavy | direct_asr | 0.1522 | 0.1538 | 6.58s | diarization_asr | 0.0000 |
| sample2_opposite_overlap | opposite | separation_asr | 0.0632 | 0.0769 | 21.26s | diarization_asr | 0.3536 |

## Diarization Order Ablation

| Sample | Overlap | Full-Audio Align CER/WER | Turn-Level ASR CER/WER | CER Change | Runtime Change | Better |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| sample2_no_overlap | none | 0.0117/0.0162 | 0.0094/0.0121 | -0.0023 | +10.14s | turn_level_asr |
| sample2_light_overlap | light | 0.0632/0.0648 | 0.0843/0.0810 | +0.0211 | +13.76s | full_audio_align |
| sample2_mid_overlap | medium | 0.1218/0.1336 | 0.2436/0.2348 | +0.1218 | +14.52s | full_audio_align |
| sample2_heavy_overlap | heavy | 0.1522/0.1538 | 0.2927/0.2753 | +0.1405 | +16.75s | full_audio_align |
| sample2_opposite_overlap | opposite | 0.4169/0.4130 | 0.5082/0.4818 | +0.0913 | +16.13s | full_audio_align |

## Detailed Results

| Sample | Overlap | Pipeline | CER | WER | Runtime | Basis | Error |
| --- | --- | --- | ---: | ---: | ---: | --- | --- |
| sample2_no_overlap | none | direct_asr | 0.0117 | 0.0162 | 8.22s | timeline |  |
| sample2_no_overlap | none | diarization_asr | 0.0117 | 0.0162 | 12.30s | speaker_block |  |
| sample2_no_overlap | none | diarization_turn_asr | 0.0094 | 0.0121 | 22.43s | speaker_block |  |
| sample2_no_overlap | none | separation_asr | 0.3443 | 0.3482 | 33.73s | speaker_block |  |
| sample2_no_overlap | none | llm_rag_refine | 0.7588 | 0.4008 | 67.25s | timeline |  |
| sample2_light_overlap | light | direct_asr | 0.0468 | 0.0486 | 7.20s | timeline |  |
| sample2_light_overlap | light | diarization_asr | 0.0632 | 0.0648 | 11.95s | speaker_block |  |
| sample2_light_overlap | light | diarization_turn_asr | 0.0843 | 0.0810 | 25.70s | speaker_block |  |
| sample2_light_overlap | light | separation_asr | 0.5293 | 0.5628 | 32.59s | speaker_block |  |
| sample2_light_overlap | light | llm_rag_refine | 0.0445 | 0.0445 | 86.28s | timeline |  |
| sample2_mid_overlap | medium | direct_asr | 0.1218 | 0.1336 | 6.96s | timeline |  |
| sample2_mid_overlap | medium | diarization_asr | 0.1218 | 0.1336 | 11.16s | speaker_block |  |
| sample2_mid_overlap | medium | diarization_turn_asr | 0.2436 | 0.2348 | 25.68s | speaker_block |  |
| sample2_mid_overlap | medium | separation_asr | 0.6393 | 0.6235 | 31.32s | speaker_block |  |
| sample2_mid_overlap | medium | llm_rag_refine | 0.1194 | 0.1296 | 85.13s | timeline |  |
| sample2_heavy_overlap | heavy | direct_asr | 0.1522 | 0.1538 | 6.58s | timeline |  |
| sample2_heavy_overlap | heavy | diarization_asr | 0.1522 | 0.1538 | 10.63s | speaker_block |  |
| sample2_heavy_overlap | heavy | diarization_turn_asr | 0.2927 | 0.2753 | 27.38s | speaker_block |  |
| sample2_heavy_overlap | heavy | separation_asr | 0.7213 | 0.7247 | 29.57s | speaker_block |  |
| sample2_heavy_overlap | heavy | llm_rag_refine | 0.1522 | 0.1538 | 74.68s | timeline |  |
| sample2_opposite_overlap | opposite | direct_asr | 0.6112 | 0.5992 | 4.67s | timeline |  |
| sample2_opposite_overlap | opposite | diarization_asr | 0.4169 | 0.4130 | 7.68s | speaker_block |  |
| sample2_opposite_overlap | opposite | diarization_turn_asr | 0.5082 | 0.4818 | 23.82s | speaker_block |  |
| sample2_opposite_overlap | opposite | separation_asr | 0.0632 | 0.0769 | 21.26s | speaker_block |  |
| sample2_opposite_overlap | opposite | llm_rag_refine | 0.6112 | 0.5992 | 92.57s | timeline |  |
