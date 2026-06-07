# 人工参考文本记录

`configs/experiment.json` 已经接入 5 个真实混合音频，但 `reference` 先保留为
`null`。请人工听写后再填入参考文本，否则 CER/WER 会基于错误答案计算，实验结论
会失真。

## 样本清单

| Sample ID | Audio | Overlap Level | Reference Status |
| --- | --- | --- | --- |
| `no_overlap` | `../xutong_code/audio_exemple/ch/chongdie/mixed_test_audio/NoOverlap.wav` | none | TODO |
| `light_overlap` | `../xutong_code/audio_exemple/ch/chongdie/mixed_test_audio/LightOverlap.wav` | light | TODO |
| `mid_overlap` | `../xutong_code/audio_exemple/ch/chongdie/mixed_test_audio/MidOverlap.wav` | medium | TODO |
| `heavy_overlap` | `../xutong_code/audio_exemple/ch/chongdie/mixed_test_audio/HeavyOverlap.wav` | heavy | TODO |
| `opposite_overlap` | `../xutong_code/audio_exemple/ch/chongdie/mixed_test_audio/OppositeOverlap.wav` | opposite | TODO |

## 建议格式

- 尽量逐字听写，不要替 ASR 或 LLM 做润色。
- 如果能区分说话人，可用 `[SPEAKER1]` 和 `[SPEAKER2]` 标注。
- 听不清的位置用 `[inaudible]`，不要猜测。
- 完成后把文本填回 `configs/experiment.json` 的 `reference` 字段。
