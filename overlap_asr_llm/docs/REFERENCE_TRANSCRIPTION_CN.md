# 人工参考文本记录

本项目目前有两套配置需要区分：

- `configs/mock.json`：快速检查配置，用于 mock/基础流程。
- `configs/base.json`：当前主实验的共享配置，已经接入
  `data/samples2/` 的 5 个音频样本，并填入整段参考文本和按说话人分块的参考文本。

当前主实验的 CER/WER、timeline 分数和 speaker-block 分数都基于
`configs/base.json` 里的 reference 计算。修改 reference
会直接影响结果，请谨慎改动。

## 当前主实验样本清单

| Sample ID | Audio | Overlap Level | Reference Status |
| --- | --- | --- | --- |
| `sample2_no_overlap` | `data/samples2/no_overlap.wav` | none | done |
| `sample2_light_overlap` | `data/samples2/light_overlap.wav` | light | done |
| `sample2_mid_overlap` | `data/samples2/mid_overlap.wav` | medium | done |
| `sample2_heavy_overlap` | `data/samples2/heavy_overlap.wav` | heavy | done |
| `sample2_opposite_overlap` | `data/samples2/opposite_overlap.wav` | opposite | done |

## 参考文本结构

当前 sample2 配置里有两类参考文本：

- `reference`：整段文本，用于 direct ASR 和 timeline/flat 比较。
- `default_reference_speakers`：按 `speaker_1` 和 `speaker_2` 分块的文本，用于
  `speaker_block_cer` 和 `speaker_block_wer`。

speaker-block 评分会自动寻找预测 speaker label 和参考 speaker label 的最佳映射，因此
`SPEAKER_00`、`SPEAKER_01`、`SPEAKER1`、`SPEAKER2` 即使顺序不同，也可以公平比较。

换句话说，speaker-block 不是按时间顺序逐句比较，而是先把同一个说话人的文本合成
一个整体 block，再和参考答案中每个说话人的 block 做比较。这样可以回答一个更适合
说话人归属的问题：模型有没有把 speaker 1 和 speaker 2 的内容分清楚？如果模型只是
把两个说话人的标签名字对调了，最佳映射会自动修正这个对调，不会把它算成严重错误。

## 建议格式

- 尽量逐字听写，不要替 ASR 或 LLM 做润色。
- 如果能区分说话人，可用 `[SPEAKER1]` 和 `[SPEAKER2]` 标注。
- 听不清的位置用 `[inaudible]`，不要猜测。
- 完成后把文本填回对应配置的 `reference` 字段；当前主实验请优先更新
  `configs/base.json`。
