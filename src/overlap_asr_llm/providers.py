"""Model provider wrappers.

The project can run in mock mode on a laptop and in real-model mode on a GPU
machine. Optional heavy dependencies are imported lazily.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import shutil


@dataclass
class Transcript:
    text: str
    segments: list[dict[str, object]]


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


class WhisperASR:
    name = "whisper"

    def __init__(self, model_name: str = "large-v3") -> None:
        import whisper

        self.model = whisper.load_model(model_name)

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


class FunASR:
    name = "funasr"

    def __init__(self) -> None:
        cache_dir = Path(
            os.environ.get("OVERLAP_ASR_LLM_CACHE_DIR", "outputs/modelscope_cache")
        ).resolve()
        cache_dir.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("MODELSCOPE_CACHE", str(cache_dir))

        from funasr import AutoModel
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model = AutoModel(
            model="iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
            vad_model="iic/speech_fsmn_vad_zh-cn-16k-common-pytorch",
            punc_model="iic/punc_ct-transformer_cn-en-common-vocab471067-large",
            device=device,
            disable_update=True,
        )

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


class MockLLMRefiner:
    name = "mock_llm_refiner"

    def refine(self, text: str, context: list[str]) -> str:
        prefix = " ".join(context[:2])
        if prefix:
            return f"{text}\n\n[Context used] {prefix}"
        return text


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


def make_diarizer(kind: str):
    if kind == "mock":
        return MockDiarizer()
    raise ValueError(f"Unsupported diarization provider: {kind}")


def make_separator(kind: str):
    if kind == "mock":
        return MockSeparator()
    if kind.startswith("sepformer"):
        parts = kind.split(":", 1)
        model_id = parts[1] if len(parts) == 2 else "speechbrain/sepformer-whamr16k"
        return SpeechBrainSeparator(model_id=model_id)
    raise ValueError(f"Unsupported separation provider: {kind}")


def make_llm_refiner(kind: str):
    if kind == "mock":
        return MockLLMRefiner()
    raise ValueError(f"Unsupported LLM provider: {kind}")
