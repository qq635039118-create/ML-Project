# 组员 Quick Start

这份说明给第一次拿到项目的组员使用。目标是：先把项目跑通，再开始分工开发。

## 当前项目状态

当前项目已经完成 mock 流程，以及服务器上的 `whisper:large-v3`
主实验结果。当前 sample2 主实验包含 5 个重叠条件和 5 条对比路径：

- `direct_asr`：`whisper:large-v3`
- `diarization_asr`：`pyannote/speaker-diarization-community-1` + `whisper:large-v3`
- `diarization_turn_asr`：先 pyannote 分说话人片段，再用 `whisper:large-v3` 转写
- `separation_asr`：`clearvoice:MossFormer2_SS_16K` + `whisper:large-v3`
- `llm_rag_refine`：基于前面 pipeline 输出的 API LLM/RAG 受限修正

主结果目录：

```text
outputs/all_pipelines/
```

主结果表：

```text
outputs/all_pipelines/run_summary.md
outputs/all_pipelines/readability_summary.md
```

GitHub 当前保留 `outputs/all_pipelines/` 中的 CSV/JSON/Markdown 结果、分离音频，
以及 `outputs/direct_asr/`、`outputs/diarization_asr/`、
`outputs/separation_asr/`、`outputs/speaker_llm_pipeline/` 这些单独 pipeline
结果目录。模型缓存、临时输出目录、无关 PDF/PPT 资料，以及
`outputs/all_pipelines/run_summary.html` 不放进 GitHub。

当前结果摘要：

- 平均 CER/WER 最好：`diarization_asr`，CER `0.1532`，WER `0.1563`。
- 平均 TRS Text 最好：`diarization_asr`，TRS Text `85.8221`。
- 平均最快：`direct_asr`，`6.72s`。
- `sample2_opposite_overlap` 例外：`separation_asr` 最好，CER `0.0632`，WER `0.0769`，TRS Text `89.0998`。
- LLM/RAG 在 light、medium、heavy 的 TRS Text 上表现好，但它主要是受限清理和格式化，不应解释为能可靠补回 ASR 漏掉的语音内容。

还没完全完成的部分：自动 overlap-aware pipeline selector、最终团队成员姓名/提交证据、正式汇报材料整合。

## 1. 拉取项目

```bash
git clone <仓库地址>
cd <仓库目录>/overlap_asr_llm
```

如果已经 clone 过：

```bash
git pull
```

## 2. 确认 Python 版本

建议使用 Python 3.10 或更高版本：

```bash
python3 --version
```

本项目的 mock 模式只依赖 Python 标准库，不需要先安装大型模型。第一次使用时安装为可编辑包：

```bash
pip install -e .
pip install pytest
```

## 3. 先跑 mock 版本

mock 模式用于检查项目结构、代码流程、输出文件是否正常。

```bash
python -m overlap_asr_llm.cli run --config configs/mock.json --mock
```

运行成功后，终端会看到类似输出：

```text
Wrote results to .../outputs
```

生成结果在：

```text
outputs/mock/results.json
outputs/mock/results.csv
outputs/mock/run_summary.md
```

## 4. 跑 smoke test

```bash
python scripts/smoke_test.py
```

这个脚本用于快速确认项目最基本功能是否能跑通。

## 5. 项目结构怎么看

```text
configs/mock.json               mock/快速检查配置
configs/base.json               当前 sample2 真实实验共享配置
configs/direct_asr.json         direct_asr 实验配置
configs/diarization_asr.json    diarization_asr 实验配置
configs/separation_asr.json     separation_asr 实验配置
configs/all_pipelines.json      最终全 pipeline 对比配置
docs/                           项目文档、实验设计、讲解稿
src/overlap_asr_llm/            核心代码
tests/                          测试代码
scripts/                        辅助脚本
outputs/                        精选实验结果；本地临时结果和模型缓存不提交
CONTRIBUTIONS.md                组员贡献记录
REPOSITORY.md                   仓库和提交说明
```

核心代码阅读顺序：

```text
config.py       读取配置
providers.py    模型封装，包括 ASR、分离、说话人标注、LLM 修正
pipelines.py    direct、diarization、separation、LLM/RAG 等实验流水线
metrics.py      CER/WER 指标
io.py           写出 JSON、CSV、Markdown 结果
cli.py          命令行入口
```

更详细的代码讲解见：

```text
docs/CORE_CODE_LINE_BY_LINE_CN.md
```

## 6. 分工开发流程

不要直接在 `main` 分支上改代码。每个人先创建自己的分支：

```bash
git checkout -b feature/<你的名字或任务名>
```

例如：

```bash
git checkout -b feature/asr-baseline
git checkout -b feature/experiment-docs
git checkout -b feature/video-script
```

完成修改后：

```bash
git status
git add <你修改的文件>
git commit -m "Describe your change"
git push -u origin feature/<你的名字或任务名>
```

然后在 GitHub 上开 Pull Request，合并到 `main`。

## 7. 提交前检查

提交前建议至少跑：

```bash
python -m overlap_asr_llm.cli run --config configs/mock.json --mock
python scripts/smoke_test.py
```

如果安装了 pytest，也可以跑：

```bash
pytest
```

如果没有安装 pytest，可以直接跑 Python 自带的 unittest：

```bash
python -m unittest discover -s tests -q
```

## 8. 不要提交哪些文件

一般不要提交本地缓存、虚拟环境、大型临时结果、无关课程资料：

```text
__pycache__/
.pytest_cache/
.venv/
venv/
outputs/caches/
outputs/<临时实验目录>/
xutong_code/
xutong_paper.pdf
AdamMaytoussi.pdf
AdamMaytoussi.pptx
AdamMaytoussi.url
*.zip
```

当前仓库保留精选输出：`outputs/all_pipelines/` 的主结果和分离音频，以及每个单独
pipeline 的结果目录。`outputs/all_pipelines/run_summary.html` 不提交。

## 9. 真模型模式

mock 跑通之后，如果要使用真实模型，需要在有合适环境的机器上安装依赖：

```bash
pip install -r requirements.txt
pip install -e .
```

要改模型，直接改对应 pipeline 的配置：

```text
configs/direct_asr.json
configs/diarization_asr.json
configs/separation_asr.json
configs/all_pipelines.json
```

可以配置的 provider 包括：

```text
ASR:        mock, whisper, whisper:<model-name>, faster-whisper, faster-whisper:<model-name>, funasr
Diarizer:   mock, pyannote, pyannote:<huggingface-model-id>
Separator:  mock, sepformer, sepformer:<huggingface-model-id>, clearvoice, clearvoice:<model-name>
LLM:        mock, api, api:<model-name>
```

`--mock` 会强制所有 provider 使用 mock，所以 smoke test 里的 LLM/RAG 是 mock。
真实 LLM/RAG 修正使用 OpenAI-compatible `api` provider，需要设置
`OPENAI_API_KEY`，也可以用 `api:<model-name>` 指定模型。

真实模型可能需要 GPU、较大下载量、API key 和额外依赖。如果只是协作写代码或文档，优先使用 mock 模式。

当前 sample2 配置已经接入 5 个真实混合音频样本，并填入整段参考文本和按说话人分块的参考文本：

```text
configs/base.json
```

运行当前真实实验的命令是：

```bash
python -m overlap_asr_llm.cli run --config configs/all_pipelines.json --incremental
```

## 10. 每个人要更新贡献记录

每位组员完成自己的部分后，记得更新：

```text
CONTRIBUTIONS.md
```

里面需要写清楚：

- 成员姓名
- 主要负责内容
- 证据，例如代码文件、文档、实验结果、视频部分
- 贡献比例

## 11. 常见问题

### 找不到 `overlap_asr_llm` 模块

如果看到类似：

```text
ModuleNotFoundError: No module named 'overlap_asr_llm'
```

说明项目还没有安装到当前 Python 环境。先安装为可编辑包：

```bash
pip install -e .
```

然后再运行：

```bash
python -m overlap_asr_llm.cli run --config configs/mock.json --mock
```

### 真实模型导入失败

如果没有安装 Whisper、FunASR、SpeechBrain 等依赖，真实模型会失败。先用 `--mock` 跑通项目流程。

### 输出结果和别人不一样

先确认是否都使用了同一个配置文件：

```text
configs/mock.json
configs/all_pipelines.json
```

当前真实实验以 `configs/all_pipelines.json` 为入口，共享内容在 `configs/base.json`。如果有人改了模型、样本、prompt 或 reference，结果会不一样。

## 最短运行命令

只想确认项目能跑时，用这一条：

```bash
python -m overlap_asr_llm.cli run --config configs/mock.json --mock
```
