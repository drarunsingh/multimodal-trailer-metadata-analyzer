"""
Inference script to predict genre, mood, and themes
from a movie trailer (local file or YouTube URL).
"""
"""
Robust YouTube → Metadata inference
Handles ALL YouTube formats safely
"""

import os
import sys
import cv2
import torch
import whisper
import numpy as np
import subprocess
import tempfile
import shutil

from transformers import VideoMAEImageProcessor, VideoMAEModel
from sentence_transformers import SentenceTransformer

# -------------------------------------------------------
# Project root
# -------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

from models.metadata_classifier import MetadataClassifier

MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "metadata_model.pt")

LABEL_NAMES = [
    "Action", "Drama", "Comedy", "Thriller",
    "Dark", "Uplifting", "Intense", "Light",
    "Revenge", "Family", "Survival",
    "Friendship", "Crime"
]

# -------------------------------------------------------
# SAFE YouTube download
# -------------------------------------------------------
def download_youtube_video(url):
    temp_dir = tempfile.mkdtemp()
    video_path = os.path.join(temp_dir, "video.mp4")

    print("[INFO] Downloading YouTube video (safe mode)...")

    cmd = [
        "yt-dlp",
        "-f", "bestvideo+bestaudio/best",
        "--merge-output-format", "mp4",
        "-o", video_path,
        url
    ]

    subprocess.run(cmd, check=True)
    return video_path, temp_dir


# -------------------------------------------------------
# Audio extraction (bulletproof)
# -------------------------------------------------------
def extract_audio(video_path, temp_dir):
    audio_path = os.path.join(temp_dir, "audio.wav")

    subprocess.run(
        [
            "ffmpeg", "-y",
            "-i", video_path,
            "-vn",
            "-ar", "16000",
            "-ac", "1",
            audio_path
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=True
    )

    return audio_path


# -------------------------------------------------------
# Video processing (360p)
# -------------------------------------------------------
def resize_to_360p(frame):
    h, w, _ = frame.shape
    scale = 360 / h
    return cv2.resize(frame, (int(w * scale), 360))


def extract_video_frames(video_path, num_frames=16):
    cap = cv2.VideoCapture(video_path)
    frames = []

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    idxs = np.linspace(0, max(total - 1, 1), num_frames).astype(int)

    for i in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, frame = cap.read()
        if ok:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(resize_to_360p(frame))

    cap.release()
    return frames


# -------------------------------------------------------
# Whisper
# -------------------------------------------------------
def transcribe_audio(audio_path):
    model = whisper.load_model("base")
    result = model.transcribe(audio_path)
    return result["text"]


# -------------------------------------------------------
# Main inference
# -------------------------------------------------------
def predict_from_youtube(url):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    video_path, temp_dir = download_youtube_video(url)
    audio_path = extract_audio(video_path, temp_dir)

    print("[INFO] Loading models...")

    video_processor = VideoMAEImageProcessor.from_pretrained("MCG-NJU/videomae-base")
    video_model = VideoMAEModel.from_pretrained("MCG-NJU/videomae-base").to(device).eval()

    text_model = SentenceTransformer("all-mpnet-base-v2")

    clf = MetadataClassifier(input_dim=1536, num_labels=len(LABEL_NAMES)).to(device)
    clf.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    clf.eval()

    print("[INFO] Transcribing audio...")
    text = transcribe_audio(audio_path)

    print("[INFO] Extracting video frames...")
    frames = extract_video_frames(video_path)

    with torch.no_grad():
        v = video_processor(frames, return_tensors="pt")
        v_emb = video_model(**v).last_hidden_state.mean(dim=1)

    t_emb = torch.tensor(text_model.encode(text), dtype=torch.float32).unsqueeze(0)
    fused = torch.cat([v_emb, t_emb], dim=1).to(device)

    with torch.no_grad():
        probs = torch.sigmoid(clf(fused)).cpu().numpy()[0]

    print("\n🎬 PREDICTED METADATA (YouTube)\n")
    for label, p in zip(LABEL_NAMES, probs):
        if p >= 0.5:
            print(f"✔ {label:<12} : {p:.2f}")

    shutil.rmtree(temp_dir)
    return probs


if __name__ == "__main__":
    url = input("Enter YouTube trailer URL: ").strip()
    predict_from_youtube(url)
