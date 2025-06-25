#  Install Dependencies
!pip install -q yt-dlp torch torchaudio transformers librosa soundfile ffmpeg-python

# Step 1: Download a 1-minute Dhoni motivational clip from YouTube
import subprocess
import os

YOUTUBE_URL = "https://www.youtube.com/watch?v=i8inTkAHJQY"  # 1-Minute Motivation: MS Dhoni – SUCCEED 1
FULL_AUDIO = "dhoni_full.wav"
CLIP_AUDIO = "dhoni_1min.wav"

# Download full video audio
if not os.path.exists(FULL_AUDIO):
    print(" Downloading full audio...")
    subprocess.run([
        "yt-dlp", "-x", "--audio-format", "wav", "--audio-quality", "0",
        "-o", FULL_AUDIO, YOUTUBE_URL
    ])

# Trim to first 60 seconds using ffmpeg
print(" Trimming to first 60 seconds...")
subprocess.run([
    "ffmpeg", "-y", "-i", FULL_AUDIO,
    "-t", "60", "-acodec", "copy", CLIP_AUDIO
])

#  Step 2: Transcribe the 1-minute clip using Wav2Vec2
import librosa
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Tokenizer

print(" Loading audio clip...")
audio, rate = librosa.load(CLIP_AUDIO, sr=16000)

print(" Loading model & tokenizer...")
tokenizer = Wav2Vec2Tokenizer.from_pretrained("facebook/wav2vec2-base-960h")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")

print(" Transcribing 1‑minute Dhoni speech...")
inputs = tokenizer(audio, return_tensors="pt", padding="longest").input_values
with torch.no_grad():
    logits = model(inputs).logits

predicted_ids = torch.argmax(logits, dim=-1)
transcription = tokenizer.batch_decode(predicted_ids)[0]

# Step 3: Show result
print("\n📄 Transcription of 1‑Minute Dhoni Speech:\n")
print(transcription)
