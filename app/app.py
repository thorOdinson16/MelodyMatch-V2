import tempfile
from pathlib import Path

import streamlit as st

from app.inference import MelodyMatchInference


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="MelodyMatch",
    page_icon="🎵",
    layout="centered",
    initial_sidebar_state="collapsed",
)


# ============================================================
# CSS
# ============================================================

st.markdown(
    """
<style>

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
}

.stApp {
    background: radial-gradient(ellipse 120% 80% at 50% -20%, #1a1d2e 0%, #0b0d14 45%, #07080c 100%);
    color: #e5e7eb;
}

/* Hide chrome */
#MainMenu, footer, header, [data-testid="stToolbar"] {
    visibility: hidden !important;
    height: 0 !important;
}

.main .block-container {
    max-width: 560px;
    padding-top: 0.9rem !important;
    padding-bottom: 0.8rem !important;
}

/* ---------- Title ---------- */
h1 {
    color: #ffffff !important;
    font-size: 2.2rem !important;
    font-weight: 800 !important;
    letter-spacing: -0.04em !important;
    text-align: center;
    margin: 0 0 0.1rem 0 !important;
    padding: 0 !important;
}

.hero-subtitle {
    text-align: center;
    color: #8b93a7;
    font-size: 0.92rem;
    line-height: 1.35;
    margin-bottom: 0.35rem;
}

.badge-row {
    display: flex;
    justify-content: center;
    margin-bottom: 0.9rem;
}

.badge {
    background: rgba(99, 102, 241, 0.1);
    border: 1px solid rgba(99, 102, 241, 0.22);
    color: #a5b4fc;
    font-size: 0.68rem;
    font-weight: 600;
    letter-spacing: 0.07em;
    text-transform: uppercase;
    padding: 0.2rem 0.7rem;
    border-radius: 999px;
}

.section-label {
    color: #6b7280;
    font-size: 0.68rem;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-bottom: 0.25rem;
}

/* ---------- Uploader ---------- */
[data-testid="stFileUploaderDropzone"] {
    background: rgba(17, 20, 30, 0.7) !important;
    border: 1.5px dashed rgba(99, 102, 241, 0.32) !important;
    border-radius: 12px !important;
    padding: 0.9rem 1rem !important;
    transition: all 0.2s ease;
}

[data-testid="stFileUploaderDropzone"]:hover {
    border-color: rgba(99, 102, 241, 0.65) !important;
    background: rgba(99, 102, 241, 0.05) !important;
}

[data-testid="stFileUploaderDropzone"] * {
    color: #9ca3af !important;
    font-size: 0.88rem !important;
}

/* ---------- Button ---------- */
.stButton > button {
    width: 100%;
    height: 2.55rem !important;
    border-radius: 10px !important;
    border: none !important;
    background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%) !important;
    color: white !important;
    font-size: 0.9rem !important;
    font-weight: 600 !important;
    box-shadow: 0 3px 14px rgba(99, 102, 241, 0.35);
    transition: all 0.2s ease !important;
    margin-top: 0.25rem !important;
}

.stButton > button:hover {
    background: linear-gradient(135deg, #5558e3 0%, #7c3aed 100%) !important;
    box-shadow: 0 5px 20px rgba(99, 102, 241, 0.45);
    transform: translateY(-1px);
}

/* ---------- HERO RESULT ---------- */
.hero-result {
    background: linear-gradient(160deg, rgba(30, 33, 50, 0.9) 0%, rgba(18, 20, 32, 0.95) 100%);
    border: 1px solid rgba(255, 255, 255, 0.07);
    border-radius: 14px;
    padding: 1rem 1.2rem 0.9rem;
    margin-top: 0.8rem;
    text-align: center;
    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.3);
    position: relative;
    overflow: hidden;
}

.hero-result::before {
    content: "";
    position: absolute;
    top: -50%;
    left: 50%;
    transform: translateX(-50%);
    width: 200px;
    height: 120px;
    background: radial-gradient(circle, rgba(99, 102, 241, 0.15) 0%, transparent 70%);
    pointer-events: none;
}

.result-label {
    color: #7c8498;
    font-size: 0.65rem;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-bottom: 0.2rem;
}

.genre-name {
    color: #ffffff;
    font-size: 1.95rem;
    font-weight: 800;
    letter-spacing: -0.03em;
    line-height: 1.1;
    margin-bottom: 0.4rem;
}

.confidence-row {
    display: inline-flex;
    align-items: center;
    gap: 0.3rem;
    background: rgba(167, 139, 250, 0.1);
    border: 1px solid rgba(167, 139, 250, 0.2);
    padding: 0.2rem 0.7rem;
    border-radius: 999px;
    font-size: 0.84rem;
    color: #c4b5fd;
    font-weight: 500;
}

.confidence-value {
    color: #e9d5ff;
    font-weight: 700;
}

/* ---------- Predictions ---------- */
.predictions-title {
    color: #d1d5db;
    font-size: 0.88rem;
    font-weight: 600;
    margin: 0.85rem 0 0.4rem 0;
}

.prediction-item {
    background: rgba(17, 20, 30, 0.55);
    border: 1px solid rgba(255, 255, 255, 0.05);
    border-radius: 10px;
    padding: 0.5rem 0.85rem;
    margin-bottom: 0.35rem;
    display: flex;
    align-items: center;
    gap: 0.7rem;
}

.rank-badge {
    width: 24px;
    height: 24px;
    border-radius: 6px;
    background: rgba(99, 102, 241, 0.15);
    color: #a5b4fc;
    font-size: 0.75rem;
    font-weight: 700;
    display: flex;
    align-items: center;
    justify-content: center;
    flex-shrink: 0;
}

.rank-badge.top {
    background: linear-gradient(135deg, #6366f1, #8b5cf6);
    color: white;
}

.pred-content {
    flex: 1;
    min-width: 0;
}

.pred-genre {
    color: #f3f4f6;
    font-size: 0.92rem;
    font-weight: 600;
    margin-bottom: 0.18rem;
}

.pred-bar-bg {
    height: 4px;
    background: rgba(255, 255, 255, 0.06);
    border-radius: 99px;
    overflow: hidden;
}

.pred-bar-fill {
    height: 100%;
    border-radius: 99px;
    background: linear-gradient(90deg, #6366f1, #a78bfa);
}

.pred-pct {
    color: #a78bfa;
    font-size: 0.88rem;
    font-weight: 600;
    min-width: 48px;
    text-align: right;
}

/* Note */
.note {
    color: #5c6474;
    font-size: 0.72rem;
    line-height: 1.35;
    margin-top: 0.7rem;
    padding-top: 0.55rem;
    border-top: 1px solid rgba(255, 255, 255, 0.05);
}

/* Footer */
.footer {
    color: #4b5563;
    text-align: center;
    font-size: 0.68rem;
    margin-top: 0.9rem;
    padding-top: 0.55rem;
    border-top: 1px solid rgba(255, 255, 255, 0.05);
}

/* Audio */
audio {
    width: 100%;
    border-radius: 10px;
    margin: 0.25rem 0 0.1rem 0;
    height: 36px;
}

/* Tighten Streamlit vertical gaps */
div[data-testid="stVerticalBlock"] > div {
    gap: 0.25rem !important;
}

</style>
""",
    unsafe_allow_html=True,
)


# ============================================================
# MODEL
# ============================================================

@st.cache_resource
def load_model():
    return MelodyMatchInference()


# ============================================================
# HEADER
# ============================================================

st.title("🎵 MelodyMatch")

st.markdown(
    """
    <div class="hero-subtitle">
        Music genre classification powered by deep learning
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="badge-row">
        <div class="badge">FMA Medium · 16 Genres · CNN</div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# UPLOAD
# ============================================================

st.markdown(
    '<div class="section-label">Audio input</div>',
    unsafe_allow_html=True,
)

uploaded_file = st.file_uploader(
    "Upload an audio file",
    type=["mp3", "wav", "flac", "ogg", "m4a"],
    label_visibility="collapsed",
)

if uploaded_file is not None:

    audio_bytes = uploaded_file.getvalue()
    st.audio(audio_bytes, format=uploaded_file.type)

    st.markdown(
        '<div class="section-label">Analysis duration</div>',
        unsafe_allow_html=True,
    )

    analysis_mode = st.radio(
        "Choose analysis duration",
        ["30 seconds", "60 seconds", "Full song"],
        horizontal=True,
        label_visibility="collapsed",
    )

    if st.button(
        "Analyze track",
        type="primary",
        use_container_width=True,
    ):
        suffix = Path(uploaded_file.name).suffix

        with tempfile.TemporaryDirectory() as temp_dir:
            source_path = Path(temp_dir) / f"source{suffix}"
            source_path.write_bytes(audio_bytes)

            try:
                model = load_model()

                with st.spinner("Analyzing..."):
                    if analysis_mode == "30 seconds":
                        predictions = model.predict(
                            str(source_path),
                            top_k=3,
                        )
                        analyzed_windows = 1

                    else:
                        from pydub import AudioSegment

                        audio = AudioSegment.from_file(str(source_path))
                        duration_ms = len(audio)
                        chunk_ms = 30_000

                        if duration_ms <= chunk_ms:
                            window_paths = [source_path]

                        elif analysis_mode == "60 seconds":
                            window_paths = []

                            for start_ms in range(
                                0,
                                min(duration_ms, 60_000),
                                chunk_ms,
                            ):
                                chunk_path = (
                                    Path(temp_dir)
                                    / f"window_{start_ms // 1000:03d}.wav"
                                )

                                audio[start_ms:start_ms + chunk_ms].export(
                                    chunk_path,
                                    format="wav",
                                )

                                window_paths.append(chunk_path)

                        else:
                            window_paths = []

                            for start_ms in range(
                                0,
                                duration_ms,
                                chunk_ms,
                            ):
                                chunk = audio[start_ms:start_ms + chunk_ms]

                                # Ignore only a very short trailing fragment.
                                if len(chunk) < 10_000:
                                    continue

                                chunk_path = (
                                    Path(temp_dir)
                                    / f"window_{start_ms // 1000:05d}.wav"
                                )

                                chunk.export(
                                    chunk_path,
                                    format="wav",
                                )

                                window_paths.append(chunk_path)

                            if not window_paths:
                                window_paths = [source_path]

                        predictions = model.predict_windows(
                            [str(path) for path in window_paths],
                            top_k=3,
                        )
                        analyzed_windows = len(window_paths)

                top = predictions[0]
                genre = top["genre"]
                probability = top["probability"] * 100

                # ---------- HERO RESULT ----------
                st.markdown(
                    f"""
                    <div class="hero-result">
                        <div class="result-label">Predicted Genre</div>
                        <div class="genre-name">{genre}</div>
                        <div class="confidence-row">
                            Model Probability
                            <span class="confidence-value">{probability:.1f}%</span>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                # ---------- TOP PREDICTIONS ----------
                st.markdown(
                    '<div class="predictions-title">Top predictions</div>',
                    unsafe_allow_html=True,
                )

                for rank, pred in enumerate(predictions, start=1):
                    genre_name = pred["genre"]
                    pct = pred["probability"] * 100
                    bar_width = min(max(pct, 3), 100)

                    rank_class = (
                        "rank-badge top"
                        if rank == 1
                        else "rank-badge"
                    )

                    st.markdown(
                        f"""
                        <div class="prediction-item">
                            <div class="{rank_class}">{rank}</div>
                            <div class="pred-content">
                                <div class="pred-genre">{genre_name}</div>
                                <div class="pred-bar-bg">
                                    <div class="pred-bar-fill"
                                         style="width: {bar_width}%;">
                                    </div>
                                </div>
                            </div>
                            <div class="pred-pct">{pct:.1f}%</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                if analysis_mode == "30 seconds":
                    analysis_text = "First 30 seconds analyzed."
                elif analysis_mode == "60 seconds":
                    analysis_text = (
                        f"{analyzed_windows} × 30-second windows analyzed."
                    )
                else:
                    analysis_text = (
                        f"{analyzed_windows} × 30-second windows analyzed "
                        "across the song."
                    )

                st.markdown(
                    f"""
                    <div class="note">
                        {analysis_text}<br>
                        Softmax outputs show relative preference among
                        16 genres — not calibrated probabilities.
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            except Exception as exc:
                st.error(f"Could not analyze this audio file: {exc}")


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">
        MelodyMatch · FMA Medium · 16 Genres
    </div>
    """,
    unsafe_allow_html=True,
)
