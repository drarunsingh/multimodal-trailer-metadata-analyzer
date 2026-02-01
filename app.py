"""
Streamlit application for multimodal trailer metadata analysis.
Accepts YouTube URLs and displays predicted metadata tags.
"""

# app.py
import streamlit as st
import numpy as np
import re
import json

from inference.predict_metadata_youtube_robust import predict_from_youtube

# --------------------------------------------
# Page config
# --------------------------------------------
st.set_page_config(
    page_title="Netflix-style Trailer Analyzer",
    page_icon="🎬",
    layout="centered"
)

# --------------------------------------------
# Header
# --------------------------------------------
st.title("🎬 Multimodal Trailer Metadata Analyzer")
st.markdown(
    """
Paste a **YouTube movie trailer link** and this system will automatically
predict **genre, mood, and thematic metadata** using **video + audio + text**.

This mirrors how modern streaming platforms enrich content for
**search, discovery, and recommendations**.
"""
)

# --------------------------------------------
# Sidebar (Netflix-style context)
# --------------------------------------------
st.sidebar.title("⚙ System Info")
st.sidebar.markdown(
    """
**Model Stack**
- VideoMAE (visual understanding)
- Whisper (audio → text)
- Sentence-BERT (language embedding)
- Multi-label classifier

**Design Choices**
- 360p resolution normalization
- Pretrained encoders
- Explainable probabilities
"""
)

# --------------------------------------------
# Input
# --------------------------------------------
youtube_url = st.text_input(
    "YouTube Trailer URL",
    placeholder="https://www.youtube.com/watch?v=XXXXXXXX"
)

threshold = st.slider(
    "Confidence Threshold",
    min_value=0.30,
    max_value=0.90,
    value=0.50,
    step=0.05,
    help="Higher threshold = stricter label selection"
)

analyze_btn = st.button("🔍 Analyze Trailer")

# --------------------------------------------
# Labels (MUST match training order)
# --------------------------------------------
LABEL_NAMES = [
    "Action", "Drama", "Comedy", "Thriller",
    "Dark", "Uplifting", "Intense", "Light",
    "Revenge", "Family", "Survival",
    "Friendship", "Crime"
]

LABEL_GROUPS = {
    "🎭 Genre": ["Action", "Drama", "Comedy", "Thriller"],
    "🎨 Mood": ["Dark", "Uplifting", "Intense", "Light"],
    "🧠 Themes": ["Revenge", "Family", "Survival", "Friendship", "Crime"]
}

# --------------------------------------------
# Utilities
# --------------------------------------------
def is_valid_youtube_url(url: str) -> bool:
    pattern = r"(https?://)?(www\.)?(youtube\.com|youtu\.be)/.+"
    return re.match(pattern, url) is not None


@st.cache_data(show_spinner=False)
def run_inference_cached(url: str):
    """Cache inference to avoid repeated downloads"""
    return predict_from_youtube(url)


# --------------------------------------------
# Inference
# --------------------------------------------
if analyze_btn:
    if not youtube_url.strip():
        st.warning("Please enter a YouTube URL.")
        st.stop()

    if not is_valid_youtube_url(youtube_url):
        st.error("Please enter a valid YouTube link.")
        st.stop()

    with st.spinner("📥 Downloading → 🎧 Transcribing → 🎥 Analyzing trailer..."):
        try:
            probs = run_inference_cached(youtube_url)
        except FileNotFoundError:
            st.error("❌ yt-dlp or ffmpeg not found in environment.")
            st.info("Install them and ensure they are in PATH.")
            st.stop()
        except Exception as e:
            st.error("❌ Inference failed.")
            st.exception(e)
            st.stop()

    st.success("✅ Analysis complete!")

    # ------------------------------------
    # Grouped results
    # ------------------------------------
    st.subheader("📊 Predicted Metadata")

    any_label = False

    for group, labels in LABEL_GROUPS.items():
        group_results = {
            label: probs[LABEL_NAMES.index(label)]
            for label in labels
            if probs[LABEL_NAMES.index(label)] >= threshold
        }

        if group_results:
            any_label = True
            st.markdown(f"### {group}")
            for label, prob in group_results.items():
                st.markdown(f"**{label}**")
                st.progress(min(float(prob), 1.0))
                st.caption(f"Confidence: {prob:.2f}")

    if not any_label:
        st.info("No labels detected above selected threshold.")

    # ------------------------------------
    # Top-K summary
    # ------------------------------------
    st.subheader("🏷 Top Predicted Tags")

    top_k = sorted(
        zip(LABEL_NAMES, probs),
        key=lambda x: x[1],
        reverse=True
    )[:5]

    for label, prob in top_k:
        st.write(f"• **{label}** — {prob:.2f}")

    # ------------------------------------
    # Export metadata
    # ------------------------------------
    metadata_json = {
        "youtube_url": youtube_url,
        "threshold": threshold,
        "predictions": {
            label: float(prob)
            for label, prob in zip(LABEL_NAMES, probs)
        }
    }

    st.download_button(
        label="⬇ Download Metadata (JSON)",
        data=json.dumps(metadata_json, indent=2),
        file_name="trailer_metadata.json",
        mime="application/json"
    )

    # ------------------------------------
    # Full probabilities
    # ------------------------------------
    with st.expander("🔎 View all probabilities"):
        for label, prob in zip(LABEL_NAMES, probs):
            st.write(f"{label:12s} : {prob:.3f}")

# --------------------------------------------
# Footer
# --------------------------------------------
st.markdown("---")
st.caption(
    "Built with VideoMAE, Whisper, Sentence-BERT & PyTorch · Netflix-style multimodal ML pipeline"
)

