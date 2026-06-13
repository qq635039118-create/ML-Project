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


def _model_cache_dir() -> Path:
    cache_dir = Path(
        os.environ.get("OVERLAP_ASR_LLM_CACHE_DIR", "outputs/caches")
    ).resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def _prepare_huggingface_download_env() -> Path:
    cache_dir = _model_cache_dir()
    os.environ.setdefault("HF_HOME", str(cache_dir / "huggingface"))
    os.environ.setdefault("HF_HUB_CACHE", str(cache_dir / "huggingface" / "hub"))
    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir / "matplotlib"))
    Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

    hf_endpoint = os.environ.get("HF_ENDPOINT", "")
    if "hf-mirror.com" in hf_endpoint:
        os.environ["HF_ENDPOINT"] = "https://huggingface.co"
    return cache_dir


class MockASR:
    name = "mock_asr"

    def transcribe(
        self,
        audio_path: Path,
        language: str,
        prompt: str | None = None,
    ) -> Transcript:
        del prompt
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
        import torch

        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = whisper.load_model(model_name)

    def transcribe(
        self,
        audio_path: Path,
        language: str,
        prompt: str | None = None,
    ) -> Transcript:
        if self.device == "cuda":
            self.model.to(self.device)
        result = self.model.transcribe(
            str(audio_path),
            language=language,
            initial_prompt=prompt,
            verbose=False,
            beam_size=1,
            best_of=1,
            temperature=0.0,
            condition_on_previous_text=False,
            fp16=True,
        )
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

    def release_gpu(self) -> None:
        if self.device != "cuda":
            return
        self.model.to("cpu")
        self.torch.cuda.empty_cache()


class FasterWhisperASR:
    name = "faster-whisper"

    def __init__(self, model_name: str = "large-v3") -> None:
        cache_dir = _prepare_huggingface_download_env()

        from faster_whisper import WhisperModel
        import torch

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.compute_type = "int8_float16" if self.device == "cuda" else "int8"
        self.model = WhisperModel(
            model_name,
            device=self.device,
            compute_type=self.compute_type,
            download_root=str(cache_dir / "faster_whisper"),
        )

    def transcribe(
        self,
        audio_path: Path,
        language: str,
        prompt: str | None = None,
    ) -> Transcript:
        segments_iter, _ = self.model.transcribe(
            str(audio_path),
            language=language,
            initial_prompt=prompt,
            beam_size=1,
            temperature=0.0,
            condition_on_previous_text=False,
            vad_filter=False,
        )
        segments = [
            {
                "start": float(segment.start),
                "end": float(segment.end),
                "text": segment.text.strip(),
                "speaker": "UNKNOWN",
            }
            for segment in segments_iter
        ]
        text = " ".join(str(segment["text"]) for segment in segments).strip()
        return Transcript(text=text, segments=segments)


class FunASR:
    name = "funasr"

    def __init__(self) -> None:
        cache_dir = Path(
            os.environ.get("MODELSCOPE_CACHE", "outputs/caches/modelscope")
        ).resolve()
        cache_dir.mkdir(parents=True, exist_ok=True)
        os.environ["MODELSCOPE_CACHE"] = str(cache_dir)

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

    def transcribe(
        self,
        audio_path: Path,
        language: str,
        prompt: str | None = None,
    ) -> Transcript:
        del language, prompt
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

    def diarize(self, audio_path: Path, speakers: int) -> list[dict[str, object]]:
        del audio_path
        speaker_count = max(speakers, 1)
        return [
            {
                "start": float(index * 2),
                "end": float((index + 1) * 2),
                "speaker": f"SPEAKER{index % speaker_count + 1}",
                "text": "",
            }
            for index in range(speaker_count)
        ]

    def label(
        self,
        transcript: Transcript,
        speakers: int,
        audio_path: Path | None = None,
    ) -> Transcript:
        del audio_path
        labeled = []
        for index, segment in enumerate(transcript.segments):
            speaker_id = index % max(speakers, 1) + 1
            labeled.append({**segment, "speaker": f"SPEAKER{speaker_id}"})
        text = " ".join(
            f"[{segment['speaker']}] {segment['text']}" for segment in labeled
        )
        return Transcript(text=text, segments=labeled)


DEFAULT_PYANNOTE_DIARIZATION_MODEL = "pyannote/speaker-diarization-community-1"


class PyannoteDiarizer:
    name = "pyannote"

    def __init__(self, model_id: str = DEFAULT_PYANNOTE_DIARIZATION_MODEL) -> None:
        token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
        if not token:
            raise RuntimeError(
                "Pyannote diarization requires HF_TOKEN. "
                "Create a Hugging Face token and accept the pyannote model terms first."
            )

        cache_dir = _prepare_huggingface_download_env()

        from pyannote.audio import Pipeline
        import torch

        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.pipeline = Pipeline.from_pretrained(
            model_id,
            token=token,
            cache_dir=cache_dir / "huggingface",
        )
        if self.pipeline is None:
            raise RuntimeError(
                f"Could not load {model_id}. Make sure your HF_TOKEN is valid "
                "and you accepted the model terms on Hugging Face."
            )
        if self.device == "cuda":
            self.pipeline.to(torch.device("cuda"))

    def diarize(self, audio_path: Path, speakers: int) -> list[dict[str, object]]:
        if self.device == "cuda":
            self.pipeline.to(self.torch.device("cuda"))
        diarization_result = self.pipeline(
            {"audio": str(audio_path)},
            num_speakers=max(speakers, 1),
        )
        diarization = self._speaker_diarization_annotation(diarization_result)
        return [
            {
                "start": float(turn.start),
                "end": float(turn.end),
                "speaker": str(label),
                "text": "",
            }
            for turn, _, label in diarization.itertracks(yield_label=True)
        ]

    def release_gpu(self) -> None:
        if self.device != "cuda":
            return
        self.pipeline.to(self.torch.device("cpu"))
        self.torch.cuda.empty_cache()

    def label(
        self,
        transcript: Transcript,
        speakers: int,
        audio_path: Path | None = None,
    ) -> Transcript:
        if audio_path is None:
            raise ValueError("Pyannote diarization requires the original audio path.")

        speaker_turns = self.diarize(audio_path, speakers)

        labeled = []
        for segment in transcript.segments:
            speaker = self._best_speaker(segment, speaker_turns)
            labeled_segment = {**segment, "speaker": speaker}
            labeled.append(labeled_segment)
            self._attach_text_to_best_turn(labeled_segment, speaker_turns)
        text = " ".join(
            f"[{segment['speaker']}] {segment['text']}" for segment in labeled
        )
        return Transcript(text=text, segments=speaker_turns)

    @staticmethod
    def _speaker_diarization_annotation(diarization_result):
        return getattr(diarization_result, "speaker_diarization", diarization_result)

    def _best_speaker(
        self,
        segment: dict[str, object],
        speaker_turns: list[dict[str, object]],
    ) -> str:
        if not speaker_turns:
            return "UNKNOWN"

        start = self._to_seconds(float(segment.get("start", 0.0)))
        end = self._to_seconds(float(segment.get("end", start)))
        if end <= start:
            end = start + 0.01

        best_label = "UNKNOWN"
        best_overlap = 0.0
        for turn in speaker_turns:
            turn_start = float(turn["start"])
            turn_end = float(turn["end"])
            overlap = max(0.0, min(end, turn_end) - max(start, turn_start))
            if overlap > best_overlap:
                best_overlap = overlap
                best_label = str(turn["speaker"])

        if best_overlap > 0:
            return best_label

        midpoint = (start + end) / 2
        nearest = min(
            speaker_turns,
            key=lambda turn: min(
                abs(midpoint - float(turn["start"])),
                abs(midpoint - float(turn["end"])),
            ),
        )
        return str(nearest["speaker"])

    def _attach_text_to_best_turn(
        self,
        segment: dict[str, object],
        speaker_turns: list[dict[str, object]],
    ) -> None:
        if not speaker_turns:
            return

        start = self._to_seconds(float(segment.get("start", 0.0)))
        end = self._to_seconds(float(segment.get("end", start)))
        if end <= start:
            return

        best_turn = None
        best_overlap = 0.0
        for turn in speaker_turns:
            overlap = max(
                0.0,
                min(end, float(turn["end"])) - max(start, float(turn["start"])),
            )
            if overlap > best_overlap:
                best_overlap = overlap
                best_turn = turn

        if best_turn is None or best_overlap <= 0:
            return

        text = str(segment.get("text", "")).strip()
        if not text:
            return
        current = str(best_turn.get("text", "")).strip()
        best_turn["text"] = f"{current} {text}".strip()

    @staticmethod
    def _to_seconds(value: float) -> float:
        return value / 1000 if value > 1000 else value


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
        cache_dir = _prepare_huggingface_download_env()
        try:
            from speechbrain.inference.separation import SepformerSeparation
        except ImportError:
            from speechbrain.pretrained import SepformerSeparation

        import torch

        self.torch = torch
        self.model = SepformerSeparation.from_hparams(
            source=model_id,
            savedir=str(cache_dir / "speechbrain" / model_id.split("/")[-1]),
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


class ClearVoiceSeparator:
    name = "clearvoice"

    def __init__(self, model_id: str = "MossFormer2_SS_16K") -> None:
        _prepare_huggingface_download_env()

        from clearvoice import ClearVoice

        self.model_id = model_id
        self.model = ClearVoice(
            task="speech_separation",
            model_names=[model_id],
        )

    def separate(self, audio_path: Path, output_dir: Path, speakers: int) -> list[Path]:
        import librosa
        import numpy as np
        import soundfile as sf

        output_dir.mkdir(parents=True, exist_ok=True)
        audio, sample_rate = sf.read(str(audio_path), always_2d=False)
        audio = np.asarray(audio, dtype=np.float32)
        if audio.ndim == 2:
            audio = audio.mean(axis=1)
        if sample_rate != 16000:
            audio = librosa.resample(y=audio, orig_sr=sample_rate, target_sr=16000)
        audio = audio.reshape(1, -1).astype(np.float32)

        separated = self.model(audio, False)
        sources = self._as_source_array(separated)
        required_sources = max(speakers, 1)
        if sources.shape[0] < required_sources:
            raise RuntimeError(
                f"ClearVoice returned {sources.shape[0]} source(s), "
                f"but {required_sources} were requested."
            )

        outputs = []
        for source_idx in range(required_sources):
            target = output_dir / f"speaker_{source_idx + 1}.wav"
            sf.write(str(target), sources[source_idx], 16000)
            outputs.append(target)
        return outputs

    @staticmethod
    def _as_source_array(separated) -> "object":
        import numpy as np

        if isinstance(separated, dict):
            if not separated:
                raise RuntimeError("ClearVoice returned an empty output dictionary.")
            separated = next(iter(separated.values()))
        elif isinstance(separated, (list, tuple)) and len(separated) == 1:
            separated = separated[0]

        sources = np.asarray(separated, dtype=np.float32)
        sources = np.squeeze(sources)
        if sources.ndim == 1:
            raise RuntimeError("ClearVoice returned one waveform instead of separated sources.")
        if sources.ndim == 3:
            sources = sources[:, 0, :]
        if sources.ndim != 2:
            raise RuntimeError(
                f"ClearVoice returned an unsupported source shape: {sources.shape}."
            )
        return sources


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
    if kind.startswith("faster-whisper"):
        parts = kind.split(":", 1)
        model_name = parts[1] if len(parts) == 2 else "large-v3"
        return FasterWhisperASR(model_name=model_name)
    if kind == "funasr":
        return FunASR()
    raise ValueError(f"Unsupported ASR provider: {kind}")


def make_diarizer(kind: str):
    if kind == "mock":
        return MockDiarizer()
    if kind.startswith("pyannote"):
        parts = kind.split(":", 1)
        model_id = parts[1] if len(parts) == 2 else DEFAULT_PYANNOTE_DIARIZATION_MODEL
        return PyannoteDiarizer(model_id=model_id)
    raise ValueError(f"Unsupported diarization provider: {kind}")


def make_separator(kind: str):
    if kind == "mock":
        return MockSeparator()
    if kind.startswith("sepformer"):
        parts = kind.split(":", 1)
        model_id = parts[1] if len(parts) == 2 else "speechbrain/sepformer-whamr16k"
        return SpeechBrainSeparator(model_id=model_id)
    if kind.startswith("clearvoice"):
        parts = kind.split(":", 1)
        model_id = parts[1] if len(parts) == 2 else "MossFormer2_SS_16K"
        return ClearVoiceSeparator(model_id=model_id)
    raise ValueError(f"Unsupported separation provider: {kind}")


def make_llm_refiner(kind: str):
    if kind == "mock":
        return MockLLMRefiner()
    raise ValueError(f"Unsupported LLM provider: {kind}")
