# Run Summary

| Sample | Overlap | Pipeline | Model | Speakers | Segments | Segments With Text | Score Basis | Primary CER | Primary WER | Speaker CER | Speaker WER | Timeline CER | Timeline WER | Runtime | Error |
| --- | --- | --- | --- | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| sample2_no_overlap | none | diarization_asr | faster-whisper+pyannote | SPEAKER_00,SPEAKER_01 | 32 | 32 | speaker_block | 0.0094 | 0.0121 | 0.0094 | 0.0121 | 0.0094 | 0.0121 | 26.4042 |  |
| sample2_light_overlap | light | diarization_asr | faster-whisper+pyannote | SPEAKER_01,SPEAKER_00 | 35 | 35 | speaker_block | 0.0843 | 0.0810 | 0.0843 | 0.0810 | 0.1077 | 0.1093 | 31.9675 |  |
| sample2_mid_overlap | medium | diarization_asr | faster-whisper+pyannote | SPEAKER_01,SPEAKER_00 | 35 | 35 | speaker_block | 0.2436 | 0.2348 | 0.2436 | 0.2348 | 0.2904 | 0.2955 | 32.7712 |  |
| sample2_heavy_overlap | heavy | diarization_asr | faster-whisper+pyannote | SPEAKER_01,SPEAKER_00 | 35 | 35 | speaker_block | 0.2927 | 0.2753 | 0.2927 | 0.2753 | 0.3794 | 0.3644 | 30.7468 |  |
| sample2_opposite_overlap | opposite | diarization_asr | faster-whisper+pyannote | SPEAKER_01,SPEAKER_00 | 36 | 36 | speaker_block | 0.5082 | 0.4818 | 0.5082 | 0.4818 | 0.9649 | 0.9717 | 28.7707 |  |
