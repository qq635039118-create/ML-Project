import os
import pandas as pd
import warnings
import torch
import librosa
import soundfile as sf
from funasr import AutoModel
import whisper
from speechbrain.pretrained import SepformerSeparation

# ===================== 配置 =====================
warnings.filterwarnings("ignore")
SAMPLE_RATE = 16000
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 固定路径
SEPARATED_AUDIO_DIR = "/root/autodl-tmp/separated_audio"
RESULTS_DIR = "/root/autodl-tmp/experiment_results"
os.makedirs(SEPARATED_AUDIO_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# 你指定的5个混合音频
MIX_AUDIOS = {
    "NoOverlap": "/root/autodl-tmp/mixed_test_audio/NoOverlap.wav",
    "LightOverlap": "/root/autodl-tmp/mixed_test_audio/LightOverlap.wav",
    "MidOverlap": "/root/autodl-tmp/mixed_test_audio/MidOverlap.wav",
    "HeavyOverlap": "/root/autodl-tmp/mixed_test_audio/HeavyOverlap.wav",
    "OppositeOverlap": "/root/autodl-tmp/mixed_test_audio/OppositeOverlap.wav"
}

# ===================== 模型加载 =====================
# ✅ 顶级音色分离：SpeechBrain 预训练SepFormer（时序掩码SOTA，效果最好）
def load_sepformer():
    model = SepformerSeparation.from_hparams(
        source="speechbrain/sepformer-whamr16k",  # 16k采样，适配中文，效果拉满
        savedir="pretrained_sepformer",
        run_opts={"device": DEVICE}
    )
    return model

# ✅ FunASR Paraformer
def load_paraformer():
    model = AutoModel(
        model="paraformer-zh",
        vad_model="fsmn-vad",
        punc_model="ct-punc",
        device=DEVICE,
        disable_update=True
    )
    return model

# ✅ Whisper
def load_whisper_model():
    return whisper.load_model("large-v3", device=DEVICE)

# ===================== 核心流程 =====================
# 1. 顶级音色/时序分离
def separate_speech(sep_model, audio_path, name):
    mix_audio, _ = librosa.load(audio_path, sr=SAMPLE_RATE)
    mix_audio = torch.tensor(mix_audio).unsqueeze(0).to(DEVICE)
    
    # SepFormer时序掩码分离（音色区分极强）
    est_sources = sep_model.separate_batch(mix_audio)
    
    # 保存2个说话人音频
    spk1 = f"{SEPARATED_AUDIO_DIR}/{name}_spk1.wav"
    spk2 = f"{SEPARATED_AUDIO_DIR}/{name}_spk2.wav"
    sf.write(spk1, est_sources[0, :, 0].cpu().numpy(), SAMPLE_RATE)
    sf.write(spk2, est_sources[0, :, 1].cpu().numpy(), SAMPLE_RATE)
    
    return spk1, spk2

# 2. 分离后转写
def transcribe(paraformer, whisper_model, spk1, spk2):
    p1 = paraformer.generate(input=spk1)[0]["text"]
    p2 = paraformer.generate(input=spk2)[0]["text"]
    para = p1 + p2

    w1 = whisper_model.transcribe(spk1, language="zh", verbose=False)["text"]
    w2 = whisper_model.transcribe(spk2, language="zh", verbose=False)["text"]
    wh = w1 + w2

    return para, wh

# ===================== 主程序 =====================
if __name__ == "__main__":
    print("加载模型：SepFormer音色分离 + Paraformer + Whisper")
    sep_model = load_sepformer()
    paraformer = load_paraformer()
    whisper_model = load_whisper_model()
    print("✅ 模型加载完成！\n")

    results = []
    for scene, path in MIX_AUDIOS.items():
        print(f"处理：{scene}")
        s1, s2 = separate_speech(sep_model, path, scene)
        para_text, whisper_text = transcribe(paraformer, whisper_model, s1, s2)
        results.append([scene, para_text, whisper_text])
        print(f"✅ {scene} 处理完成\n")

    # 保存结果
    df = pd.DataFrame(results, columns=["重叠场景", "音色分离后_Paraformer", "音色分离后_Whisper"])
    df.to_excel(f"{RESULTS_DIR}/best_separation_results.xlsx", index=False)


    print(f"分离音频：{SEPARATED_AUDIO_DIR}")
    print(f"结果表格：{RESULTS_DIR}/best_separation_results.xlsx")
    for scene, path in MIX_AUDIOS.items():
        print(f"处理：{scene}")
        
        # 步骤1：时序掩码分离
        spk1, spk2 = separate_speech(sepformer, path, scene)
        
        # 步骤2：分离后音频转写
        para_text, whisper_text = transcribe(paraformer, whisper_model, spk1, spk2)
        
        results.append([scene, para_text, whisper_text])
        print(f"{scene} 处理完成\n")

    # 保存结果
    df = pd.DataFrame(results, columns=[
        "重叠场景", 
        "时序分离后_Paraformer", 
        "时序分离后_Whisper"
    ])
    df.to_excel(f"{RESULTS_DIR}/sepformer_results.xlsx", index=False)

    print("全部任务完成！")
    print(f"分离音频：{SEPARATED_AUDIO_DIR}")
    print(f"转写结果：{RESULTS_DIR}/sepformer_results.xlsx")