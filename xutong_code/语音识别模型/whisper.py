import torch
import whisper
import warnings
import numpy as np
import librosa
from speechbrain.inference.speaker import SpeakerRecognition
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ====================== 配置 ======================
AUDIO_PATH = "/root/autodl-tmp/audio_files/zhixun.mp4"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ====================== 自动选最佳说话人数量（3-8人） ======================
def get_best_k(features, min_k=3, max_k=8):  
    if len(features) <= 1:
        return min_k
    
    scaler = StandardScaler()
    X = scaler.fit_transform(features)
    best_k = min_k
    best_score = -1

    # 遍历范围：3 ~ 8
    for k in range(min_k, min(max_k + 1, len(X))):
        try:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init="auto")
            labels = kmeans.fit_predict(X)
            score = silhouette_score(X, labels)
            if score > best_score:
                best_score = score
                best_k = k
        except:
            continue
    return best_k

# ====================== 主程序 ======================
if __name__ == "__main__":
    print("=" * 70)
    print(" WHISPER large-v3 + SPEECHBRAIN 说话人聚类（高精度版）")
    print("=" * 70)
    print(f"音频：{AUDIO_PATH.split('/')[-1]}")
    print(f"设备：{DEVICE}")
    print("=" * 70)

    # 加载 Whisper 模型
    model = whisper.load_model("large-v3", device=DEVICE)

    result = model.transcribe(
        AUDIO_PATH,
        language="zh",
        verbose=False,
        temperature=0.1,
        condition_on_previous_text=False,
        word_timestamps=True
    )

    segments = result["segments"]
    print(f"\n语音片段数：{len(segments)}")
    print("正在用 SpeechBrain 提取说话人特征...")

    # ====================== SpeechBrain 说话人编码器 ======================
    encoder = SpeakerRecognition.from_hparams(
        source="speechbrain/spkrec-xvect-voxceleb",
        savedir="tmp_speechbrain",
        run_opts={"device": DEVICE}
    )

    # 加载音频
    y, sr = librosa.load(AUDIO_PATH, sr=16000)
    embeddings = []

    for seg in segments:
        start = int(seg["start"] * sr)
        end = int(seg["end"] * sr)
        segment_wav = y[start:end]

        # 提取 SpeechBrain 说话人向量
        with torch.no_grad():
            emb = encoder.encode_batch(torch.tensor(segment_wav).unsqueeze(0).to(DEVICE))
            emb = emb.squeeze().cpu().numpy()
        embeddings.append(emb)

    # 自动判断最佳说话人数量 3~8
    best_k = get_best_k(embeddings, min_k=3, max_k=8)
    print(f"最佳说话人数量：{best_k}")

    # 聚类
    kmeans = KMeans(n_clusters=best_k, random_state=42)
    labels = kmeans.fit_predict(embeddings)

    # ====================== 输出带说话人的文本 ======================
    print("\n【最终识别结果】")
    print("=" * 70)
    for i, seg in enumerate(segments):
        spk_id = labels[i] + 1
        print(f"[SPEAKER{spk_id}] {seg['text'].strip()}")