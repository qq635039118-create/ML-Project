import os
import numpy as np
import pandas as pd
import librosa
import soundfile as sf
import warnings
import whisper
import torch
import torchaudio
from funasr import AutoModel
from sklearn.cluster import AgglomerativeClustering
from speechbrain.inference.separation import SepformerSeparation as separator

# 全局配置
warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
SAMPLE_RATE = 16000
SEPARATION_SAMPLE_RATE = 8000

# 路径配置
MIXED_WAV_PATH = " "
SEPARATED_AUDIO_DIR = " "
RESULTS_DIR = " "

# 创建必要文件夹
for d in [SEPARATED_AUDIO_DIR, RESULTS_DIR]:
    os.makedirs(d, exist_ok=True)

# 加载已转好的混合WAV音频
def load_mixed_audio(wav_path):
    print(f"正在加载已转换的混合音频: {wav_path}")
    y, sr = librosa.load(wav_path, sr=SAMPLE_RATE)
    print(f"音频加载完成，时长: {len(y)/sr:.2f}秒")
    return y, wav_path


def load_separation_model(device):
    print("正在加载SpeechBrain SepFormer-wsj02mix分离模型...")
    return separator.from_hparams(
        source="speechbrain/sepformer-wsj02mix",
        savedir="/root/.cache/speechbrain/sepformer-wsj02mix",
        run_opts={"device": device}
    )

# 自动重采样到8kHz
def separate_speakers(mixed_path, separation_model):
    print("正在分离双说话人音频...")
    
    # 1. 加载原始16kHz音频并重采样到8kHz
    y, sr = librosa.load(mixed_path, sr=SEPARATION_SAMPLE_RATE)
    
    # 2. 保存临时8kHz文件供模型处理
    temp_8k_path = os.path.join(SEPARATED_AUDIO_DIR, "temp_mixed_8k.wav")
    sf.write(temp_8k_path, y, SEPARATION_SAMPLE_RATE)
    
    # 3. 执行分离（官方标准调用）
    est_sources = separation_model.separate_file(path=temp_8k_path)
    
    # 4. 提取两个说话人音频并保存（输出为8kHz）
    spk1 = est_sources[:, :, 0].detach().cpu().numpy().squeeze()
    spk2 = est_sources[:, :, 1].detach().cpu().numpy().squeeze()
    
    spk1_path = os.path.join(SEPARATED_AUDIO_DIR, "zx_spk1_sepformer_8k.wav")
    spk2_path = os.path.join(SEPARATED_AUDIO_DIR, "zx_spk2_sepformer_8k.wav")
    
    sf.write(spk1_path, spk1, SEPARATION_SAMPLE_RATE)
    sf.write(spk2_path, spk2, SEPARATION_SAMPLE_RATE)
    
    # 删除临时文件
    os.remove(temp_8k_path)
    
    print(f"SepFormer语音分离完成（输出8kHz）")
    print(f"  说话人1: {spk1_path}")
    print(f"  说话人2: {spk2_path}")
    
    return spk1, spk2, spk1_path, spk2_path

# 加载FunASR语音识别模型
def load_funasr():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModel(
        model="paraformer-zh",
        model_revision="v2.0.4",
        vad_model="fsmn-vad",
        punc_model="ct-punc",
        device=device,
        disable_update=True
    )
    return model, device

# 加载CAM++说话人嵌入模型
def load_speaker_model(device):
    return AutoModel(
        model="iic/speech_campplus_sv_zh-cn_16k-common",
        device=device,
        disable_update=True
    )

# 加载Whisper模型
def load_whisper(device):
    return whisper.load_model("large-v3", device=device)

# 修复版转写+自动说话人标注
def transcribe_and_label(audio_path, funasr_model, speaker_model, n_speakers=2):
    res = funasr_model.generate(input=audio_path, batch_size=1, return_raw=True)[0]
    raw_text = res["text"]
    
    # 兼容没有sentences字段的情况
    if "sentences" in res:
        segments = res["sentences"]
    else:
        segments = [{
            "text": raw_text,
            "start": 0.0,
            "end": len(librosa.load(audio_path, sr=SAMPLE_RATE)[0]) / SAMPLE_RATE
        }]
    
    # 单说话人模式直接返回
    if n_speakers == 1:
        return raw_text, raw_text
    
    # 只加载一次音频
    y, sr = librosa.load(audio_path, sr=SAMPLE_RATE)
    embeddings = []
    valid_segments = []
    
    for seg in segments:
        start = int(seg["start"] * sr)
        end = int(seg["end"] * sr)
        seg_audio = y[start:end]
        
        # 跳过太短的片段
        if len(seg_audio) < int(0.1 * sr):
            continue
        
        # 提取说话人特征
        sv_res = speaker_model.generate(input=seg_audio, batch_size=1)[0]
        
        # 兼容不同版本的返回字段
        if "embedding" in sv_res:
            emb = sv_res["embedding"]
        elif "embeddings" in sv_res:
            emb = sv_res["embeddings"][0]
        else:
            emb = list(sv_res.values())[0]
        
        # 强制压缩为一维数组
        emb = np.squeeze(np.array(emb))
        if emb.ndim != 1:
            emb = emb.flatten()
        
        embeddings.append(emb)
        valid_segments.append(seg)
    
    # 防止没有有效片段
    if len(embeddings) < 2:
        return raw_text, f"[SPEAKER1] {raw_text}"
    
    # 转换为numpy数组
    embeddings = np.array(embeddings)
    
    # 层次聚类
    clustering = AgglomerativeClustering(n_clusters=n_speakers, metric="cosine", linkage="average")
    labels = clustering.fit_predict(embeddings)
    
    # 生成带标注文本
    labeled_text = ""
    for i, seg in enumerate(valid_segments):
        labeled_text += f"[SPEAKER{labels[i]+1}] {seg['text']} "
    
    return raw_text, labeled_text.strip()

# 生成四个流程全部文本
def process_all_flows(mix_path, spk1_path, spk2_path, funasr_model, speaker_model, whisper_model):
    # 流程1：混合音频直接转写（基线）
    print("🔄 处理流程1: 混合音频直接转写")
    flow1_funasr, _ = transcribe_and_label(mix_path, funasr_model, speaker_model, n_speakers=1)
    flow1_whisper = whisper_model.transcribe(mix_path, language="zh", verbose=False)["text"]
    
    # 流程2：混合音频转写+自动说话人标注
    print("🔄 处理流程2: 混合转写+自动说话人标注")
    flow2_funasr, flow2_labeled = transcribe_and_label(mix_path, funasr_model, speaker_model, n_speakers=2)
    
    # 流程3：分离后分别转写+合并标注
    print("🔄 处理流程3: 分离转写+合并标注")
    spk1_raw, _ = transcribe_and_label(spk1_path, funasr_model, speaker_model, n_speakers=1)
    spk2_raw, _ = transcribe_and_label(spk2_path, funasr_model, speaker_model, n_speakers=1)
    flow3_labeled = f"[SPEAKER1] {spk1_raw} [SPEAKER2] {spk2_raw}"
    
    # 流程4：分离后转写+预留LLM修正位置
    flow4_spk1_raw = spk1_raw
    flow4_spk2_raw = spk2_raw
    flow4_merged_raw = flow3_labeled
    
    return (
        flow1_funasr, flow1_whisper,
        flow2_funasr, flow2_labeled,
        spk1_raw, spk2_raw, flow3_labeled,
        flow4_spk1_raw, flow4_spk2_raw, flow4_merged_raw
    )

# 主程序
if __name__ == "__main__":
    # 1. 加载已转好的混合WAV
    mixed_audio, mixed_path = load_mixed_audio(MIXED_WAV_PATH)
    
    # 2. 加载所有模型
    funasr_model, device = load_funasr()
    separation_model = load_separation_model(device)
    speaker_model = load_speaker_model(device)
    whisper_model = load_whisper(device)
    
    # 3. 分离双说话人
    spk1_audio, spk2_audio, spk1_path, spk2_path = separate_speakers(
        mixed_path, separation_model
    )
    
    # 4. 生成四个流程全部文本
    print("\n===== 开始生成四流程实验文本 =====")
    (
        flow1_funasr, flow1_whisper,
        flow2_funasr, flow2_labeled,
        flow3_spk1, flow3_spk2, flow3_labeled,
        flow4_spk1, flow4_spk2, flow4_merged
    ) = process_all_flows(
        mixed_path, spk1_path, spk2_path,
        funasr_model, speaker_model, whisper_model
    )
    
    # 5. 保存所有结果到Excel
    results = []
    results.append([
        "真实辩论质询场景",
        # 流程1
        flow1_funasr, flow1_whisper,
        # 流程2
        flow2_funasr, flow2_labeled,
        # 流程3
        flow3_spk1, flow3_spk2, flow3_labeled,
        # 流程4
        flow4_spk1, flow4_spk2, flow4_merged,
        "", "", ""  # 预留LLM修正列
    ])
    
    df = pd.DataFrame(results, columns=[
        "场景名称",
        # 流程1：混合直接转写（基线）
        "流程1_FunASR原始转写", "流程1_Whisper原始转写",
        # 流程2：混合转写+自动标注
        "流程2_FunASR原始转写", "流程2_自动说话人标注",
        # 流程3：分离转写+合并标注
        "流程3_说话人1原始转写", "流程3_说话人2原始转写", "流程3_合并标注文本",
        # 流程4：预留LLM修正
        "流程4_说话人1原始转写", "流程4_说话人2原始转写", "流程4_合并原始文本",
        "流程4_说话人1修正后", "流程4_说话人2修正后", "流程4_合并修正后"
    ])
    
    excel_path = os.path.join(RESULTS_DIR, "真实辩论四流程实验结果.xlsx")
    df.to_excel(excel_path, index=False)
    
    print(f"📁 混合音频: {mixed_path}")
    print(f"📁 分离后说话人1: {spk1_path}")
    print(f"📁 分离后说话人2: {spk2_path}")
    print(f"📁 完整实验结果: {excel_path}")