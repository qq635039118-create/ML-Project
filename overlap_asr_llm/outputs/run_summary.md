# Run Summary

| Sample | Overlap | Pipeline | Model | CER | WER | Runtime | Error |
| --- | --- | --- | --- | ---: | ---: | ---: | --- |
| no_overlap | none | direct_asr | whisper | 0.0235 | 1.0000 | 36.7055 |  |
| no_overlap | none | diarization_asr | whisper+mock_diarizer | 0.4930 | 20.0000 | 28.6665 |  |
| no_overlap | none | separation_asr | mock_separator+whisper | 1.0939 | 4.0000 | 33.5318 |  |
| no_overlap | none | llm_rag_refine | mock_llm_refiner | 4.4742 | 57.0000 | 0.0916 |  |
| light_overlap | light | direct_asr | whisper | 0.1972 | 1.0000 | 26.9534 |  |
| light_overlap | light | diarization_asr | whisper+mock_diarizer | 0.6103 | 18.0000 | 28.2024 |  |
| light_overlap | light | separation_asr | mock_separator+whisper | 0.9765 | 4.0000 | 33.6734 |  |
| light_overlap | light | llm_rag_refine | mock_llm_refiner | 4.0610 | 55.0000 | 0.0856 |  |
| mid_overlap | medium | direct_asr | whisper | 0.3474 | 1.0000 | 25.1953 |  |
| mid_overlap | medium | diarization_asr | whisper+mock_diarizer | 0.6432 | 16.0000 | 25.8380 |  |
| mid_overlap | medium | separation_asr | mock_separator+whisper | 0.8732 | 4.0000 | 32.2356 |  |
| mid_overlap | medium | llm_rag_refine | mock_llm_refiner | 3.7653 | 53.0000 | 0.0781 |  |
| heavy_overlap | heavy | direct_asr | whisper | 0.4131 | 1.0000 | 22.8817 |  |
| heavy_overlap | heavy | diarization_asr | whisper+mock_diarizer | 0.6385 | 10.0000 | 22.8386 |  |
| heavy_overlap | heavy | separation_asr | mock_separator+whisper | 0.6808 | 4.0000 | 29.6229 |  |
| heavy_overlap | heavy | llm_rag_refine | mock_llm_refiner | 3.1033 | 47.0000 | 0.0657 |  |
| opposite_overlap | opposite | direct_asr | whisper | 0.8779 | 1.0000 | 24.6050 |  |
| opposite_overlap | opposite | diarization_asr | whisper+mock_diarizer | 1.2160 | 22.0000 | 24.9235 |  |
| opposite_overlap | opposite | separation_asr | mock_separator+whisper | 0.9249 | 4.0000 | 31.9656 |  |
| opposite_overlap | opposite | llm_rag_refine | mock_llm_refiner | 3.9202 | 59.0000 | 0.0812 |  |
