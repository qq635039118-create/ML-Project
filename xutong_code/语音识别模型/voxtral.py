import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import torch
import time
import librosa
from transformers import AutoModel, AutoProcessor
from jiwer import wer, cer

# ==================== 配置 ====================
AUDIO_PATH = ""
# TRUE_TEXT = """ """

MODEL = "mistralai/Voxtral-Mini-4B-Realtime-2602"

# ==================== 加载 ====================
print("加载 Voxtral Mini 4B 官方模型...")
processor = AutoProcessor.from_pretrained(MODEL)
model = AutoModel.from_pretrained(
    MODEL,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True
)

# ==================== 音频处理 ====================
y, sr = librosa.load(AUDIO_PATH, sr=16000)
inputs = processor(
    audio=y,
    text="Transcribe the audio.",
    return_tensors="pt"
).to("cuda", dtype=torch.bfloat16)  

# ==================== 推理 ====================
print("推理中...")
t0 = time.time()

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=2048
    )

text = processor.decode(outputs[0], skip_special_tokens=True)
cost = time.time() - t0

# ==================== 结果 ====================
print("="*70)
print("VOXTRAL MINI 4B 运行成功！")
print(f"耗时：{cost:.2f}s")
print(f"结果：{text}")
# print(f"WER：{wer(TRUE_TEXT, text):.4f}")
# print("="*70)