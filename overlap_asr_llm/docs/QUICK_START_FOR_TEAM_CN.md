# 组员 Quick Start

这份说明给第一次拿到项目的组员使用。目标是：先把项目跑通，再开始分工开发。

## 当前项目状态

当前项目已经完成 mock 流程和三条真实非 LLM pipeline 的 sample2 实验：

- `direct_asr`：`faster-whisper:large-v3`
- `diarization_asr`：`pyannote/speaker-diarization-community-1` + `faster-whisper:large-v3`
- `separation_asr`：`clearvoice:MossFormer2_SS_16K` + `faster-whisper:large-v3`

主结果目录：

```text
outputs/all_pipelines/
```

主结果表：

```text
outputs/all_pipelines/run_summary.md
```

还没完全完成的部分：真实 LLM/RAG 修正、自动 overlap-aware pipeline selector、当前 sample2 结果的完整人工可读性评分。

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
outputs/                        运行生成的结果
CONTRIBUTIONS.md                组员贡献记录
REPOSITORY.md                   仓库和提交说明
```

核心代码阅读顺序：

```text
config.py       读取配置
providers.py    模型封装，包括 ASR、分离、说话人标注、LLM 修正
pipelines.py    四条实验流水线
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

一般不要提交本地缓存、虚拟环境、大型临时结果：

```text
__pycache__/
.pytest_cache/
.venv/
venv/
outputs/<临时实验目录>/
*.zip
```

如果最终报告需要引用某个小型结果目录，可以保留精选输出，例如当前
`outputs/all_pipelines/`。

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
LLM:        mock
```

真实模型可能需要 GPU、较大下载量和额外依赖。如果只是协作写代码或文档，优先使用 mock 模式。

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
