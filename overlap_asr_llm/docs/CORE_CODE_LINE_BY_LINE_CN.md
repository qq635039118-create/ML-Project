# 核心代码对应讲解

这份文档用“对应代码 + 中文解释”的方式讲项目核心代码。阅读顺序建议是：

1. `config.py`：读取实验配置。
2. `providers.py`：封装 ASR、说话人标注、语音分离、LLM 修正。
3. `pipelines.py`：组织 direct、diarization、turn-level diarization、separation、LLM/RAG 等实验流水线。
4. `metrics.py`：计算 CER/WER。
5. `io.py`：写出结果文件。
6. `cli.py`：命令行入口。

## `config.py`：配置怎么变成 Python 对象

```python
@dataclass(frozen=True)
class Sample:
    id: str
    audio_path: Path
    overlap_level: str
    speakers: int
    reference: str | None = None
```

这段定义“一个音频样本”的结构。比如配置文件里每个 sample 都会变成一个 `Sample` 对象。

- `id`：样本名字，例如 `no_overlap_demo`。
- `audio_path`：音频路径。
- `overlap_level`：重叠程度，例如 `none`、`light`、`heavy`。
- `speakers`：说话人数。
- `reference`：人工参考文本；有它才能算 CER/WER。
- `frozen=True`：创建后不允许随便改字段，避免实验配置运行中被误改。

```python
@dataclass(frozen=True)
class ExperimentConfig:
    project_name: str
    output_dir: Path
    language: str
    models: dict[str, str]
    rag_context: list[str]
    samples: list[Sample]
    base_dir: Path
```

这段定义“整个实验配置”的结构。

- `project_name`：项目名。
- `output_dir`：结果输出目录。
- `language`：语音语言，比如 `zh`。
- `models`：每个模块用什么模型，例如 ASR 用 `mock`。
- `rag_context`：给 LLM/RAG 用的背景知识。
- `samples`：所有样本。
- `base_dir`：项目根目录，用来把相对路径转成绝对路径。

```python
def load_config(path: str | Path) -> ExperimentConfig:
    config_path = Path(path).resolve()
    base_dir = config_path.parent.parent
    with config_path.open("r", encoding="utf-8") as f:
        raw: dict[str, Any] = json.load(f)
```

这段开始读取 JSON 配置。

- `Path(path).resolve()`：把传入路径变成绝对路径。
- `config_path.parent.parent`：如果配置在 `configs/mock.json`，上一级的上一级就是项目根目录。
- `json.load(f)`：把 JSON 文件读成 Python 字典 `raw`。

```python
    required = ["project_name", "output_dir", "language", "models", "samples"]
    missing = [key for key in required if key not in raw]
    if missing:
        raise ValueError(f"Missing required config keys: {', '.join(missing)}")
```

这段做基础校验。

- `required`：列出必须存在的配置项。
- `missing`：找出 JSON 里缺少的项。
- 如果缺配置，就直接报错，不继续跑实验。

```python
    samples = [
        Sample(
            id=item["id"],
            audio_path=(base_dir / item["audio_path"]).resolve(),
            overlap_level=item.get("overlap_level", "unknown"),
            speakers=int(item.get("speakers", 1)),
            reference=item.get("reference"),
        )
        for item in raw["samples"]
    ]
```

这段把 JSON 里的样本列表转换成 `Sample` 对象列表。

- `item["id"]`：样本必须有 id。
- `(base_dir / item["audio_path"]).resolve()`：把相对音频路径变成绝对路径。
- `item.get("overlap_level", "unknown")`：没有写重叠程度就用 `unknown`。
- `int(item.get("speakers", 1))`：没有写说话人数就默认 1。
- `item.get("reference")`：参考文本可选。

```python
    return ExperimentConfig(
        project_name=raw["project_name"],
        output_dir=(base_dir / raw["output_dir"]).resolve(),
        language=raw["language"],
        models=dict(raw["models"]),
        rag_context=list(raw.get("rag_context", [])),
        samples=samples,
        base_dir=base_dir,
    )
```

这段返回完整实验配置。

简单说：`load_config()` 的任务就是把 `configs/mock.json` 变成后面代码能直接使用的 `ExperimentConfig`。

## `providers.py`：模型封装

### 通用转写结果

```python
@dataclass
class Transcript:
    text: str
    segments: list[dict[str, object]]
```

项目里所有 ASR 都统一返回 `Transcript`。

- `text`：完整转写文本。
- `segments`：分段结果，每段通常包含 `start`、`end`、`text`、`speaker`。

### Mock ASR

```python
class MockASR:
    name = "mock_asr"

    def transcribe(self, audio_path: Path, language: str) -> Transcript:
        stem = audio_path.stem.replace("_", " ")
        text = f"mock transcript for {stem}"
        return Transcript(
            text=text,
            segments=[
                {"start": 0.0, "end": 2.0, "text": text, "speaker": "UNKNOWN"}
            ],
        )
```

这是假的 ASR，用于本地测试。

- `audio_path.stem`：取音频文件名，不带扩展名。
- `replace("_", " ")`：把文件名里的下划线换成空格。
- `text = ...`：构造一段模拟转写文本。
- 返回的 `Transcript` 里只有一个分段，时间从 0 到 2 秒，说话人是 `UNKNOWN`。

### Whisper ASR

```python
class WhisperASR:
    name = "whisper"

    def __init__(self, model_name: str = "large-v3") -> None:
        import whisper

        self.model = whisper.load_model(model_name)
```

这段负责加载 Whisper。

注意 `import whisper` 写在 `__init__` 里，而不是文件顶部。这样 mock 模式运行时不需要安装 Whisper。

```python
    def transcribe(self, audio_path: Path, language: str) -> Transcript:
        result = self.model.transcribe(str(audio_path), language=language, verbose=False)
        segments = [
            {
                "start": float(seg.get("start", 0.0)),
                "end": float(seg.get("end", 0.0)),
                "text": str(seg.get("text", "")).strip(),
                "speaker": "UNKNOWN",
            }
            for seg in result.get("segments", [])
        ]
        return Transcript(text=str(result.get("text", "")).strip(), segments=segments)
```

这段真正调用 Whisper 转写。

- `self.model.transcribe(...)`：调用 Whisper。
- `str(audio_path)`：Whisper 接收字符串路径。
- `language=language`：告诉模型语言。
- `verbose=False`：不打印详细过程。
- `segments = [...]`：把 Whisper 原始分段转成项目统一格式。
- `speaker` 固定为 `UNKNOWN`，因为 Whisper 本身不负责说话人识别。

### FunASR

```python
class FunASR:
    name = "funasr"

    def __init__(self) -> None:
        from funasr import AutoModel

        self.model = AutoModel(
            model="paraformer-zh",
            vad_model="fsmn-vad",
            punc_model="ct-punc",
            disable_update=True,
        )
```

这段加载 FunASR 中文模型。

- `paraformer-zh`：中文 ASR 主模型。
- `fsmn-vad`：语音活动检测模型，用来判断哪里有人声。
- `ct-punc`：标点恢复模型。
- `disable_update=True`：避免运行时自动更新。

```python
    def transcribe(self, audio_path: Path, language: str) -> Transcript:
        del language
        result = self.model.generate(input=str(audio_path), batch_size=1, return_raw=True)[0]
        text = str(result.get("text", "")).strip()
        raw_segments = result.get("sentences") or [
            {"start": 0.0, "end": 0.0, "text": text}
        ]
        segments = [
            {
                "start": float(segment.get("start", 0.0)),
                "end": float(segment.get("end", 0.0)),
                "text": str(segment.get("text", "")).strip(),
                "speaker": "UNKNOWN",
            }
            for segment in raw_segments
        ]
        return Transcript(text=text, segments=segments)
```

这段调用 FunASR 转写。

- `del language`：FunASR 这里没有使用传入语言，显式删掉表示“我知道它没用”。
- `generate(...)`：执行识别。
- `text`：完整识别文本。
- `raw_segments`：优先使用 FunASR 返回的句子分段；如果没有，就自己造一个分段。
- `segments`：统一成项目格式。

### Mock Diarizer

```python
class MockDiarizer:
    name = "mock_diarizer"

    def label(self, transcript: Transcript, speakers: int) -> Transcript:
        labeled = []
        for index, segment in enumerate(transcript.segments):
            speaker_id = index % max(speakers, 1) + 1
            labeled.append({**segment, "speaker": f"SPEAKER{speaker_id}"})
        text = " ".join(
            f"[{segment['speaker']}] {segment['text']}" for segment in labeled
        )
        return Transcript(text=text, segments=labeled)
```

这是假的说话人标注器。

- 它不会真的识别谁在说话。
- 它只是按分段顺序轮流分配 `SPEAKER1`、`SPEAKER2`。
- `{**segment, "speaker": ...}`：复制原分段，并覆盖 `speaker` 字段。
- 最终文本会变成 `[SPEAKER1] xxx [SPEAKER2] yyy` 这种格式。

### Mock Separator

```python
class MockSeparator:
    name = "mock_separator"

    def separate(self, audio_path: Path, output_dir: Path, speakers: int) -> list[Path]:
        output_dir.mkdir(parents=True, exist_ok=True)
        outputs = []
        for speaker_idx in range(1, max(speakers, 1) + 1):
            target = output_dir / f"{audio_path.stem}_speaker{speaker_idx}.wav"
            if audio_path.exists():
                shutil.copyfile(audio_path, target)
            else:
                target.touch()
            outputs.append(target)
        return outputs
```

这是假的语音分离器。

- 它不会真的把多人声音分开。
- 它只是为每个说话人复制一份原音频。
- 如果原音频不存在，就创建空文件，保证流程能跑通。

### SpeechBrain SepFormer

```python
class SpeechBrainSeparator:
    name = "sepformer"

    def __init__(self, model_id: str = "speechbrain/sepformer-whamr16k") -> None:
        try:
            from speechbrain.inference.separation import SepformerSeparation
        except ImportError:
            from speechbrain.pretrained import SepformerSeparation

        import torch

        self.torch = torch
        self.model = SepformerSeparation.from_hparams(
            source=model_id,
            savedir=f"pretrained_{model_id.split('/')[-1]}",
            run_opts={"device": "cuda" if torch.cuda.is_available() else "cpu"},
        )
```

这段加载真实的 SepFormer 语音分离模型。

- 先尝试新版 SpeechBrain 导入路径。
- 如果失败，再尝试旧版导入路径。
- `torch.cuda.is_available()`：有 GPU 就用 GPU，否则用 CPU。
- `savedir=...`：指定预训练模型缓存目录。

```python
    def separate(self, audio_path: Path, output_dir: Path, speakers: int) -> list[Path]:
        import soundfile as sf

        output_dir.mkdir(parents=True, exist_ok=True)
        sources = self.model.separate_file(path=str(audio_path))
        if hasattr(sources, "detach"):
            sources = sources.detach().cpu()
        if len(sources.shape) == 3:
            sources = sources[0]

        outputs = []
        source_count = min(int(sources.shape[-1]), max(speakers, 1))
        for speaker_idx in range(source_count):
            target = output_dir / f"{audio_path.stem}_speaker{speaker_idx + 1}.wav"
            sf.write(target, sources[:, speaker_idx].numpy(), 16000)
            outputs.append(target)
        return outputs
```

这段执行真实语音分离。

- `separate_file(...)`：把混合音频分成多个声源。
- `detach().cpu()`：如果结果是 PyTorch tensor，就搬到 CPU，方便写文件。
- `sources[0]`：去掉 batch 维度。
- `source_count`：实际输出数量不能超过模型输出，也不能超过配置中的说话人数。
- `sf.write(...)`：把每个声源写成 wav 文件。

### Mock LLM Refiner

```python
class MockLLMRefiner:
    name = "mock_llm_refiner"

    def refine(self, text: str, context: list[str]) -> str:
        prefix = " ".join(context[:2])
        if prefix:
            return f"{text}\n\n[Context used] {prefix}"
        return text
```

这是假的 LLM 修正器。

- 它不会真的改写文本。
- 它只是把前两条 RAG 背景知识附加到文本后面。
- 作用是让 LLM/RAG 流水线在没有真实 LLM 的情况下也能跑通。

### Provider 工厂函数

```python
def make_asr(kind: str):
    if kind == "mock":
        return MockASR()
    if kind.startswith("whisper"):
        parts = kind.split(":", 1)
        model_name = parts[1] if len(parts) == 2 else "large-v3"
        return WhisperASR(model_name=model_name)
    if kind == "funasr":
        return FunASR()
    raise ValueError(f"Unsupported ASR provider: {kind}")
```

这段根据配置创建 ASR。

- `"mock"` 创建 `MockASR`。
- `"whisper"` 创建默认 Whisper `large-v3`。
- `"whisper:base"` 这种写法会加载指定 Whisper 模型。
- `"faster-whisper"` 或 `"faster-whisper:large-v3"` 使用 faster-whisper。
- `"funasr"` 创建 FunASR。
- 其他值直接报错。

```python
def make_diarizer(kind: str):
    if kind == "mock":
        return MockDiarizer()
    if kind.startswith("pyannote"):
        ...
    if kind.startswith("speechbrain"):
        ...
    raise ValueError(f"Unsupported diarization provider: {kind}")
```

当前说话人标注支持 mock、pyannote 和 SpeechBrain。pyannote 更适合主实验的
speaker diarization，SpeechBrain 主要用于 speaker embedding/聚类式对比。

```python
def make_separator(kind: str):
    if kind == "mock":
        return MockSeparator()
    if kind.startswith("sepformer"):
        parts = kind.split(":", 1)
        model_id = parts[1] if len(parts) == 2 else "speechbrain/sepformer-whamr16k"
        return SpeechBrainSeparator(model_id=model_id)
    if kind.startswith("clearvoice"):
        ...
    raise ValueError(f"Unsupported separation provider: {kind}")
```

这段根据配置创建语音分离器。

- `"mock"` 创建假分离器。
- `"sepformer"` 使用默认 SepFormer。
- `"sepformer:xxx"` 使用指定 Hugging Face 模型 id。
- `"clearvoice"` 或 `"clearvoice:xxx"` 使用 ClearVoice 分离模型。

```python
def make_llm_refiner(kind: str):
    if kind == "mock":
        return MockLLMRefiner()
    if kind == "api":
        return ApiLLMRefiner()
    if kind.startswith("api:"):
        return ApiLLMRefiner(model_name=kind.split(":", 1)[1])
    raise ValueError(f"Unsupported LLM provider: {kind}")
```

当前 LLM 修正支持 mock 和 OpenAI-compatible API。`api` 会使用环境变量中的
`OPENAI_API_KEY`，`api:<model-name>` 可以指定模型。

## `pipelines.py`：实验流水线

### 结果结构

```python
@dataclass
class PipelineResult:
    sample_id: str
    audio_path: str
    overlap_level: str
    pipeline: str
    model: str
    text: str
    speaker_labels: str
    runtime_seconds: float
    cer: float | None
    wer: float | None
    error: str | None = None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)
```

这段定义每条流水线输出什么。

- 前几个字段记录样本信息。
- `pipeline`：是哪条流水线。
- `model`：用了哪个模型或模型组合。
- `text`：输出文本。
- `speaker_labels`：说话人标签。
- `runtime_seconds`：耗时。
- `cer` / `wer`：错误率。
- `error`：如果失败，错误写在这里。
- `to_dict()`：方便写 JSON/CSV。

### 评分和标签辅助函数

```python
def _score(reference: str | None, text: str) -> tuple[float | None, float | None]:
    if not reference:
        return None, None
    return cer(reference, text), wer(reference, text)
```

如果没有参考文本，就不计算 CER/WER；如果有，就调用 `metrics.py` 里的 `cer()` 和 `wer()`。

```python
def _labels_from_segments(segments: list[dict[str, object]]) -> str:
    labels = [str(segment.get("speaker", "UNKNOWN")) for segment in segments]
    return ",".join(dict.fromkeys(labels))
```

这段从分段里提取说话人标签。

- `segment.get("speaker", "UNKNOWN")`：没有 speaker 就用 `UNKNOWN`。
- `dict.fromkeys(labels)`：去重，同时保留首次出现顺序。
- 最后用逗号拼成字符串。

```python
def _error_result(
    sample: Sample,
    pipeline: str,
    model: str,
    started: float,
    error: Exception,
) -> PipelineResult:
    return PipelineResult(
        sample_id=sample.id,
        audio_path=str(sample.audio_path),
        overlap_level=sample.overlap_level,
        pipeline=pipeline,
        model=model,
        text="",
        speaker_labels="",
        runtime_seconds=round(time.perf_counter() - started, 4),
        cer=None,
        wer=None,
        error=f"{type(error).__name__}: {error}",
    )
```

这段统一处理错误。

如果某条流水线失败，项目不会整体崩溃，而是返回一个带 `error` 的 `PipelineResult`。

### 流水线 1：直接 ASR

```python
def run_direct_asr(config: ExperimentConfig, sample: Sample) -> PipelineResult:
    started = time.perf_counter()
    model_name = config.models.get("asr", "mock")
    try:
        asr = make_asr(model_name)
        transcript = asr.transcribe(sample.audio_path, config.language)
        cer_value, wer_value = _score(sample.reference, transcript.text)
        return PipelineResult(
            sample_id=sample.id,
            audio_path=str(sample.audio_path),
            overlap_level=sample.overlap_level,
            pipeline="direct_asr",
            model=asr.name,
            text=transcript.text,
            speaker_labels=_labels_from_segments(transcript.segments),
            runtime_seconds=round(time.perf_counter() - started, 4),
            cer=cer_value,
            wer=wer_value,
        )
    except Exception as exc:
        return _error_result(sample, "direct_asr", model_name, started, exc)
```

这条流水线最简单：直接把混合音频丢给 ASR。

流程是：

1. 读取 ASR 模型名。
2. 用 `make_asr()` 创建 ASR。
3. 对原音频转写。
4. 如果有参考文本，计算 CER/WER。
5. 返回结果。
6. 如果中途失败，返回错误结果。

### 流水线 2：ASR + 说话人标注

```python
def run_diarization_asr(config: ExperimentConfig, sample: Sample) -> PipelineResult:
    started = time.perf_counter()
    asr_name = config.models.get("asr", "mock")
    diarizer_name = config.models.get("diarization", "mock")
    model_name = f"{asr_name}+{diarizer_name}"
    try:
        asr = make_asr(asr_name)
        diarizer = make_diarizer(diarizer_name)
        transcript = asr.transcribe(sample.audio_path, config.language)
        labeled = diarizer.label(transcript, sample.speakers)
        cer_value, wer_value = _score(sample.reference, labeled.text)
        return PipelineResult(
            sample_id=sample.id,
            audio_path=str(sample.audio_path),
            overlap_level=sample.overlap_level,
            pipeline="diarization_asr",
            model=f"{asr.name}+{diarizer.name}",
            text=labeled.text,
            speaker_labels=_labels_from_segments(labeled.segments),
            runtime_seconds=round(time.perf_counter() - started, 4),
            cer=cer_value,
            wer=wer_value,
        )
    except Exception as exc:
        return _error_result(sample, "diarization_asr", model_name, started, exc)
```

这条流水线先 ASR，再给分段加说话人标签。

它的目标是比较：只做 ASR 和“ASR 后带 speaker 标签”哪个更有用。

### 流水线 3：语音分离 + ASR

```python
def run_separation_asr(config: ExperimentConfig, sample: Sample) -> PipelineResult:
    started = time.perf_counter()
    asr_name = config.models.get("asr", "mock")
    separator_name = config.models.get("separation", "mock")
    model_name = f"{separator_name}+{asr_name}"
    try:
        asr = make_asr(asr_name)
        separator = make_separator(separator_name)
        separated_dir = config.output_dir / "separated_audio" / sample.id
        speaker_paths = separator.separate(sample.audio_path, separated_dir, sample.speakers)
        parts = []
        segments = []
        for index, speaker_path in enumerate(speaker_paths, start=1):
            transcript = asr.transcribe(speaker_path, config.language)
            speaker = f"SPEAKER{index}"
            parts.append(f"[{speaker}] {transcript.text}")
            segments.extend({**segment, "speaker": speaker} for segment in transcript.segments)
        text = " ".join(parts)
        cer_value, wer_value = _score(sample.reference, text)
        return PipelineResult(
            sample_id=sample.id,
            audio_path=str(sample.audio_path),
            overlap_level=sample.overlap_level,
            pipeline="separation_asr",
            model=f"{separator.name}+{asr.name}",
            text=text,
            speaker_labels=_labels_from_segments(segments),
            runtime_seconds=round(time.perf_counter() - started, 4),
            cer=cer_value,
            wer=wer_value,
        )
    except Exception as exc:
        return _error_result(sample, "separation_asr", model_name, started, exc)
```

这条流水线先把混合音频按说话人分离，再分别 ASR。

核心步骤：

1. 创建 ASR。
2. 创建 separator。
3. 把分离后的音频写到 `outputs/separated_audio/<sample_id>/`。
4. 对每个分离音频分别转写。
5. 给每个转写文本加 `[SPEAKER1]`、`[SPEAKER2]` 标签。
6. 合并成最终文本。

### 流水线 4：LLM/RAG 修正

```python
def run_llm_rag_refine(
    config: ExperimentConfig,
    sample: Sample,
    source_results: list[PipelineResult],
) -> PipelineResult:
    started = time.perf_counter()
    llm_name = config.models.get("llm", "mock")
    try:
        refiner = make_llm_refiner(llm_name)
        source_text = "\n".join(
            f"{result.pipeline}: {result.text}"
            for result in source_results
            if result.sample_id == sample.id and not result.error
        )
        refined = refiner.refine(source_text, config.rag_context)
        cer_value, wer_value = _score(sample.reference, refined)
        return PipelineResult(
            sample_id=sample.id,
            audio_path=str(sample.audio_path),
            overlap_level=sample.overlap_level,
            pipeline="llm_rag_refine",
            model=refiner.name,
            text=refined,
            speaker_labels="LLM_REFINED",
            runtime_seconds=round(time.perf_counter() - started, 4),
            cer=cer_value,
            wer=wer_value,
        )
    except Exception as exc:
        return _error_result(sample, "llm_rag_refine", llm_name, started, exc)
```

这条流水线不是直接处理音频，而是处理前面流水线产生的文本。

它会：

1. 收集同一个样本的已有成功结果。
2. 拼成 `source_text`。
3. 把 `source_text` 和 `rag_context` 交给 LLM refiner。
4. 输出修正文本。

当前 mock 版本只是把上下文附在文本后面。

### 总入口：运行全部流水线

```python
def run_all(config: ExperimentConfig) -> list[PipelineResult]:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    results: list[PipelineResult] = []
    for sample in config.samples:
        direct = run_direct_asr(config, sample)
        diarized = run_diarization_asr(config, sample)
        separated = run_separation_asr(config, sample)
        results.extend([direct, diarized, separated])
        results.append(run_llm_rag_refine(config, sample, results))
    return results
```

这是整个实验的核心调度函数。

对每个样本，它都会按顺序跑：

1. `direct_asr`
2. `diarization_asr`
3. `diarization_turn_asr`（如果配置中启用）
4. `separation_asr`
5. `llm_rag_refine`

最后返回所有结果。

## `metrics.py`：CER/WER 怎么算

```python
def normalize_text(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text
```

这段做文本清洗。

- 全部转小写。
- 去掉首尾空白。
- 多个空格、换行、tab 压成一个空格。

```python
def edit_distance(left: list[str], right: list[str]) -> int:
    previous = list(range(len(right) + 1))
    for i, left_item in enumerate(left, start=1):
        current = [i]
        for j, right_item in enumerate(right, start=1):
            cost = 0 if left_item == right_item else 1
            current.append(
                min(
                    previous[j] + 1,
                    current[j - 1] + 1,
                    previous[j - 1] + cost,
                )
            )
        previous = current
    return previous[-1]
```

这是编辑距离，也就是把一个序列变成另一个序列最少要几步。

三种操作：

- 删除：`previous[j] + 1`
- 插入：`current[j - 1] + 1`
- 替换或匹配：`previous[j - 1] + cost`

`cost` 为 0 表示两个字符/词相同；为 1 表示不同。

```python
def cer(reference: str, hypothesis: str) -> float:
    ref = list(normalize_text(reference).replace(" ", ""))
    hyp = list(normalize_text(hypothesis).replace(" ", ""))
    if not ref:
        return 0.0 if not hyp else 1.0
    return edit_distance(ref, hyp) / len(ref)
```

CER 是字符错误率。

- `reference`：正确答案。
- `hypothesis`：模型输出。
- 去掉空格后按字符比较。
- 错误率 = 字符编辑距离 / 参考文本字符数。

```python
def wer(reference: str, hypothesis: str) -> float:
    ref = tokenize_words(reference)
    hyp = tokenize_words(hypothesis)
    if not ref:
        return 0.0 if not hyp else 1.0
    return edit_distance(ref, hyp) / len(ref)
```

WER 是词错误率。

- 和 CER 类似，但比较单位从“字符”变成“词”。
- 英文主要按空格切词。
- 中文会优先用 `jieba` 分词；如果环境里没有 `jieba`，就退回到简单 token 匹配。

### speaker-block CER/WER 是什么意思

普通 `timeline_cer` 会把分段按时间顺序拼起来比较，适合看“整段时间线文本像不像参考答案”。
但说话人标注任务还关心另一个问题：模型有没有把两个人各自说的话分清楚。

speaker-block 的做法是：

1. 参考答案里有按说话人分块的文本，例如 `speaker_1` 一整段、`speaker_2` 一整段。
2. 模型输出里也把文本按预测 speaker label 分组，例如 `SPEAKER_00`、`SPEAKER_01`。
3. 评分时不会假设 `SPEAKER_00` 一定等于 `speaker_1`，而是尝试所有合理映射。
4. 选择 CER/WER 最低的映射，作为 `speaker_block_cer` 和 `speaker_block_wer`。

这样做是为了公平。很多 diarization 模型的 speaker label 是任意编号，可能这次把第一个人叫
`SPEAKER_00`，下次叫 `SPEAKER_01`。如果两个说话人的内容分得对，只是 label 名字反了，
speaker-block 评分不会把它当成严重错误。

一个简化例子：

```text
参考:
speaker_1: 今天必须上线
speaker_2: 不能牺牲用户体验

模型:
SPEAKER_00: 不能牺牲用户体验
SPEAKER_01: 今天必须上线
```

如果强行按名字比，两个 speaker 都像错了；但 speaker-block 会发现最佳映射是
`SPEAKER_00 -> speaker_2`、`SPEAKER_01 -> speaker_1`，所以主要评价文本内容和说话人归属是否正确。

## `io.py`：结果怎么写到文件

```python
FIELDNAMES = [
    "sample_id",
    "audio_path",
    "overlap_level",
    "pipeline",
    "model",
    "text",
    "speaker_labels",
    "runtime_seconds",
    "cer",
    "wer",
    "error",
]
```

这段定义 CSV 字段顺序，也就是结果表每一列叫什么。

```python
def write_results(results: list[PipelineResult], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = [result.to_dict() for result in results]

    with (output_dir / "results.json").open("w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)

    with (output_dir / "results.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    write_summary(results, output_dir / "run_summary.md")
```

这段一次性写三种输出。

- `results.json`：完整结构化结果。
- `results.csv`：方便用 Excel 或 pandas 看。
- `run_summary.md`：方便直接读的 Markdown 表格。

```python
def write_summary(results: list[PipelineResult], path: Path) -> None:
    lines = [
        "# Run Summary",
        "",
        "| Sample | Overlap | Pipeline | Model | CER | WER | Runtime | Error |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | --- |",
    ]
    for result in results:
        cer_value = "" if result.cer is None else f"{result.cer:.4f}"
        wer_value = "" if result.wer is None else f"{result.wer:.4f}"
        error = result.error or ""
        lines.append(
            "| "
            f"{result.sample_id} | "
            f"{result.overlap_level} | "
            f"{result.pipeline} | "
            f"{result.model} | "
            f"{cer_value} | "
            f"{wer_value} | "
            f"{result.runtime_seconds:.4f} | "
            f"{error} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
```

这段生成 Markdown 汇总表。

- CER/WER 如果是 `None`，就留空。
- 否则保留 4 位小数。
- 每个 `PipelineResult` 变成表格中的一行。

## `cli.py`：命令行怎么启动项目

```python
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run overlapping speech experiments.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run all experiment pipelines.")
    run_parser.add_argument(
        "--config",
        default="configs/mock.json",
        help="Path to the JSON experiment config.",
    )
    run_parser.add_argument(
        "--mock",
        action="store_true",
        help="Force all providers to mock mode for local smoke tests.",
    )
    return parser
```

这段定义命令行参数。

项目支持这样的命令：

```bash
python -m overlap_asr_llm.cli run --config configs/mock.json --mock
```

其中：

- `run`：运行实验。
- `--config`：指定配置文件。
- `--mock`：强制所有 provider 使用 mock 模式。

```python
def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "run":
        config = load_config(Path(args.config))
        if args.mock:
            config.models.update(
                {"asr": "mock", "diarization": "mock", "separation": "mock", "llm": "mock"}
            )
        results = run_all(config)
        write_results(results, config.output_dir)
        print(f"Wrote results to {config.output_dir}")
        return 0

    parser.error(f"Unsupported command: {args.command}")
    return 2
```

这段是命令行主逻辑。

执行 `run` 时：

1. 读取配置。
2. 如果有 `--mock`，把所有模型强制改成 mock。
3. 调用 `run_all(config)` 跑全部流水线。
4. 调用 `write_results(...)` 写结果。
5. 返回 0 表示成功。

```python
if __name__ == "__main__":
    raise SystemExit(main())
```

这段表示：当你直接运行这个文件或用 `python -m overlap_asr_llm.cli` 时，执行 `main()`。

`SystemExit(main())` 会把 `main()` 返回的数字作为程序退出码。

## `__init__.py`：包信息

```python
"""Overlapping speech ASR + LLM experiment package."""

__version__ = "0.1.0"
```

这个文件让 `overlap_asr_llm` 成为一个 Python 包，并记录当前版本号。

## 配置文件和代码的对应关系

```json
{
  "models": {
    "asr": "mock",
    "diarization": "mock",
    "separation": "mock",
    "llm": "mock"
  }
}
```

这段配置会对应到：

```python
asr = make_asr("mock")
diarizer = make_diarizer("mock")
separator = make_separator("mock")
refiner = make_llm_refiner("mock")
```

也就是所有模型都用本地 mock 版本。

```json
{
  "samples": [
    {
      "id": "no_overlap_demo",
      "audio_path": "data/samples/no_overlap_demo.wav",
      "overlap_level": "none",
      "speakers": 2,
      "reference": "speaker one introduces the task speaker two explains the result"
    }
  ]
}
```

这段 sample 会对应到：

```python
Sample(
    id="no_overlap_demo",
    audio_path=Path(".../data/samples/no_overlap_demo.wav"),
    overlap_level="none",
    speakers=2,
    reference="speaker one introduces the task speaker two explains the result",
)
```

然后 `run_all(config)` 会按配置对这个样本依次跑启用的流水线。

## 整体运行链路

```text
cli.py
  -> load_config()
      -> ExperimentConfig / Sample
  -> run_all()
      -> run_direct_asr()
      -> run_diarization_asr()
      -> run_separation_asr()
      -> run_llm_rag_refine()
  -> write_results()
      -> results.json
      -> results.csv
      -> run_summary.md
```

一句话总结：

这个项目先从 JSON 读实验配置，然后为每个音频样本跑四种处理方案，计算指标，最后把结果写到输出目录。
