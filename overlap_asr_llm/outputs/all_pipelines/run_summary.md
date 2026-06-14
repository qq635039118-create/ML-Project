# Run Summary

| Sample | Overlap | Pipeline | Model | Score Basis | Primary CER | Primary WER | Speaker CER | Speaker WER | Timeline CER | Timeline WER | Runtime | Error |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| sample2_no_overlap | none | direct_asr | faster-whisper | timeline | 0.0117 | 0.0162 |  |  | 0.0117 | 0.0162 | 44.5689 |  |
| sample2_no_overlap | none | diarization_asr | faster-whisper+pyannote | speaker_block | 0.0094 | 0.0121 | 0.0094 | 0.0121 | 0.0094 | 0.0121 | 213.7214 |  |
| sample2_no_overlap | none | separation_asr | clearvoice+faster-whisper | speaker_block | 0.1756 | 0.2065 | 0.1756 | 0.2065 | 0.1710 | 0.1943 | 439.0200 |  |
| sample2_light_overlap | light | direct_asr | faster-whisper | timeline | 0.0468 | 0.0486 |  |  | 0.0468 | 0.0486 | 53.6878 |  |
| sample2_light_overlap | light | diarization_asr | faster-whisper+pyannote | speaker_block | 0.0843 | 0.0810 | 0.0843 | 0.0810 | 0.1077 | 0.1093 | 256.2663 |  |
| sample2_light_overlap | light | separation_asr | clearvoice+faster-whisper | speaker_block | 0.5855 | 0.6275 | 0.5855 | 0.6275 | 0.5340 | 0.5709 | 341.8456 |  |
