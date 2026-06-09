# Run Summary

## funasr

| Sample | Overlap | Pipeline | Model | CER | WER | Runtime | Error |
| --- | --- | --- | --- | ---: | ---: | ---: | --- |
| new_no_overlap | none | direct_asr | funasr | 0.0398 | 0.0784 | 1.2530 |  |
| new_light_overlap | light | direct_asr | funasr | 0.2587 | 0.2941 | 0.5143 |  |
| new_mid_overlap | medium | direct_asr | funasr | 0.4030 | 0.4510 | 0.4692 |  |
| new_heavy_overlap | heavy | direct_asr | funasr | 0.5871 | 0.6667 | 0.4050 |  |
| new_opposite_overlap | opposite | direct_asr | funasr | 0.8259 | 0.9020 | 0.4866 |  |
| xutong_no_overlap | none | direct_asr | funasr | 0.8386 | 0.8580 | 0.5838 |  |
| xutong_light_overlap | light | direct_asr | funasr | 0.8228 | 0.8284 | 0.5664 |  |
| xutong_mid_overlap | medium | direct_asr | funasr | 0.8038 | 0.8047 | 0.5171 |  |
| xutong_heavy_overlap | heavy | direct_asr | funasr | 0.8070 | 0.8580 | 0.4537 |  |
| xutong_opposite_overlap | opposite | direct_asr | funasr | 0.7785 | 0.8284 | 0.3478 |  |

## whisper:large-v3

| Sample | Overlap | Pipeline | Model | CER | WER | Runtime | Error |
| --- | --- | --- | --- | ---: | ---: | ---: | --- |
| new_no_overlap | none | direct_asr | whisper | 0.0149 | 0.0294 | 98.5009 |  |
| new_light_overlap | light | direct_asr | whisper | 0.2289 | 0.2647 | 100.5313 |  |
| new_mid_overlap | medium | direct_asr | whisper | 0.3433 | 0.3725 | 85.6358 |  |
| new_heavy_overlap | heavy | direct_asr | whisper | 0.4229 | 0.4706 | 73.5770 |  |
| new_opposite_overlap | opposite | direct_asr | whisper | 0.8657 | 0.9020 | 95.8197 |  |
| xutong_no_overlap | none | direct_asr | whisper | 0.7848 | 0.7219 | 141.9710 |  |
| xutong_light_overlap | light | direct_asr | whisper | 0.7690 | 0.7337 | 130.0931 |  |
| xutong_mid_overlap | medium | direct_asr | whisper | 0.7595 | 0.7278 | 123.5642 |  |
| xutong_heavy_overlap | heavy | direct_asr | whisper | 0.7057 | 0.7041 | 116.9519 |  |
| xutong_opposite_overlap | opposite | direct_asr | whisper | 0.7627 | 0.7574 | 87.3271 |  |

