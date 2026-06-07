"""Model provider wrappers.

The project can run in mock mode on a laptop and in real-model mode on a GPU
machine. Optional heavy dependencies are imported lazily.
"""

from __future__ import annotations
from sklearn.metrics import silhouette_score
from dataclasses import dataclass
from pathlib import Path
import shutil
import torch



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
        from funasr import AutoModel

        self.model = AutoModel(
            model="paraformer-zh",
            vad_model="fsmn-vad",
            punc_model="ct-punc",
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

    def label(self, transcript: Transcript,audio_path: Path,speakers: int) -> Transcript:
        labeled = []
        for index, segment in enumerate(transcript.segments):
            speaker_id = index % max(speakers, 1) + 1
            labeled.append({**segment, "speaker": f"SPEAKER{speaker_id}"})
        text = " ".join(
            f"[{segment['speaker']}] {segment['text']}" for segment in labeled
        )
        return Transcript(text=text, segments=labeled)
    

class SpeechBrainDiarizer:
    name = "funasr_campp_diarizer"

    def __init__(self, model_id: str = "damo/speech_eres2net_sv_zh-cn_16k-common") -> None:
        import os
        os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"
        
        import torch
        from modelscope.pipelines import pipeline
        from modelscope.utils.constant import Tasks
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"\n[INFO] Loading Alibaba SV Pipeline: {model_id} on {self.device}...")
        
        self.sv_pipeline = pipeline(
            task=Tasks.speaker_verification,
            model=model_id,
            model_revision='v1.0.5'
        )

        try:
            if self.device == "cuda" and hasattr(self.sv_pipeline, 'model'):
                self.sv_pipeline.model.to(self.device)
                print("[INFO] Successfully moved ModelScope model to CUDA.")
        except Exception as e:
            print(f"[WARN] Failed to move model to device: {e}, running with default device.")



    def label(self, transcript: Transcript, audio_path: Path, speakers: int) -> Transcript:
        import soundfile as sf
        import torch
        import numpy as np
        from sklearn.cluster import AgglomerativeClustering
        from sklearn.preprocessing import StandardScaler
        from sklearn.metrics import silhouette_score

        # 1. 使用 soundfile 库稳健加载原始音频
        data, fs = sf.read(str(audio_path))
        
        if len(data.shape) > 1:
            data = data.mean(axis=1)
        
        if fs != 16000:
            import torchaudio
            signal_tensor = torch.FloatTensor(data).unsqueeze(0)
            signal_tensor = torchaudio.transforms.Resample(orig_freq=fs, new_freq=16000)(signal_tensor)
            data = signal_tensor.squeeze(0).numpy()

        embeddings = []
        valid_segments = []

        # 2. 纯内存特征提取
        for segment in transcript.segments:
            start_sample = int(segment["start"] * 16000)
            end_sample = int(segment["end"] * 16000)

            if end_sample - start_sample < 16000:
                continue

            segment_data = data[start_sample:end_sample]
            
            if len(segment_data) == 0 or np.max(np.abs(segment_data)) == 0:
                continue
                
            segment_data = segment_data.astype(np.float32)
            
            try:
                # 喂给阿里中括号包裹的 NumPy 数组
                sv_result = self.sv_pipeline([segment_data])
                
                if isinstance(sv_result, list):
                    sv_result = sv_result[0]
                
                # 1. 自动适配多键名
                emb = None
                for key in ['spk_embeddings', 'spk_embedding', 'embedding', 'embeddings']:
                    if key in sv_result:
                        emb = sv_result[key]
                        break
                
                if emb is None:
                    continue
                
                # 将 Tensor 或 ndarray 彻底压回纯一维 (192,) 向量
                # 无论阿里套了多少层中括号，或者是 PyTorch Tensor，通通强转为一维 NumPy 数组
                if hasattr(emb, "cpu"):
                    emb_np = emb.squeeze().cpu().numpy().flatten()
                else:
                    emb_np = np.array(emb).squeeze().flatten()
                
                # 3. 安全防污染检查：如果声纹向量里包含 nan、inf 或者全为 0，直接丢弃
                if emb_np is None or len(emb_np) == 0 or np.isnan(emb_np).any() or np.isinf(emb_np).any():
                    print("[WARN] Extracted embedding contains NaN/Inf, skipping this segment.")
                    continue
                    
                # 确保提取出的维度是绝对正确的 192 维
                if emb_np.shape[0] == 192:
                    embeddings.append(emb_np)
                    valid_segments.append(segment)
                else:
                    print(f"[WARN] Unexpected embedding shape: {emb_np.shape}, expected (192,)")

            except Exception as e:
                import traceback
                print(f"[WARN] Failed to extract embedding: {e}")
                traceback.print_exc()
                continue

        if not embeddings:
            return transcript

        # 3. 特征标准化
        X = np.array(embeddings)
        X = StandardScaler().fit_transform(X)

        # 4. 层次聚类 (自适应盲聚类)
        max_possible_speakers = 6
        best_n_clusters = 1
        best_labels = np.zeros(len(X), dtype=int)
        best_score = -1.0

        if len(X) >= 2:
            max_clusters_to_try = min(max_possible_speakers, len(X))
            if max_clusters_to_try >= 2:
                for n_clusters in range(2, max_clusters_to_try + 1):
                    clustering = AgglomerativeClustering(n_clusters=n_clusters, metric='euclidean', linkage='ward')
                    labels = clustering.fit_predict(X)
                    
                    score = silhouette_score(X, labels)
                    if score > best_score:
                        best_score = score
                        best_n_clusters = n_clusters
                        best_labels = labels

            if best_score < 0.1:
                best_n_clusters = 1
                best_labels = np.zeros(len(X), dtype=int)

        # 5. 将聚类标签写回
        for i, segment in enumerate(valid_segments):
            segment["speaker"] = f"SPEAKER_{best_labels[i] + 1}"

        text = " ".join(f"[{seg['speaker']}] {seg['text']}" for seg in valid_segments)
        print(f"\n[SUCCESS] Adaptive Clustering Done. Detected {best_n_clusters} speakers in meeting.")

        return Transcript(text=text, segments=valid_segments)

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

class LocalLLMRefiner:
    name = "local_llm_refiner"

    def __init__(self, model_name: str = "qwen2.5:7b"):
        from openai import OpenAI
        
        # 本地运行Ollama
        self.client = OpenAI(
            base_url='http://localhost:11434/v1',
            api_key='ollama',
        )
        self.model_name = model_name

    def refine(self, text: str, context: list[str]) -> str:
        rag_bullet_points = "\n".join(f"- {c}" for c in context)
        system_prompt = (
            "你是一个专业的语音转录文本格式化工具。你的任务是根据提供的RAG词汇表，"
            "修正文本中的错别字，并输出带有说话人标签的对话剧本。\n"
            f"【专业词汇表】\n{rag_bullet_points}\n\n"
            "【要求】只输出最终剧本，严禁编造音频中没有的内容，不要输出任何解释说明。"
        )

        try:
            print(f"[INFO] Requesting Local Model ({self.model_name}) via Ollama...")
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"整理以下ASR输出:\n\n{text}"}
                ],
                temperature=0.0 # 对小模型来说，温度设为0能最大程度减少幻觉
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"[WARN] Local LLM Failed: {e}")
            return f"[LLM ERROR] {text}"

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
    if kind == "speechbrain":
        return SpeechBrainDiarizer()
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
    if kind.startswith("ollama"):
        parts = kind.split(":", 1)
        model_name = parts[1] if len(parts) == 2 else "qwen2.5:7b"
        return LocalLLMRefiner(model_name=model_name)
    raise ValueError(f"Unsupported LLM provider: {kind}")
