"""
ASL Finger Spelling Recognition - Fully Automatic
Real-time ASL to speech with AI-powered word correction.
"""

import time
import cv2
import numpy as np
import pickle
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration, WebRtcMode
import av
import threading
from collections import deque
from pathlib import Path
import mediapipe as mp

from utils.tts_handler import speak_letter, speak_word
from utils.openai_helper import get_word_corrector

# OpenAI API Key
OPENAI_API_KEY = "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"


# Page config
st.set_page_config(
    page_title="ASL Recognition",
    page_icon="🤟",
    layout="wide"
)

# Modern CSS
st.markdown("""
<style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    .main .block-container {
        padding-top: 2rem;
        max-width: 1400px;
    }

    .header-box {
        background: linear-gradient(135deg, #1e3a5f 0%, #2d5a87 100%);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        color: white;
        text-align: center;
    }

    .word-display {
        background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%);
        color: white;
        font-size: 3rem;
        font-weight: 700;
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        letter-spacing: 6px;
        min-height: 100px;
        box-shadow: 0 8px 32px rgba(59, 130, 246, 0.4);
        margin: 2rem 0;
    }

    .correction-box {
        background: linear-gradient(135deg, #8b5cf6 0%, #6d28d9 100%);
        color: white;
        font-size: 2rem;
        font-weight: 600;
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        margin: 1rem 0;
        box-shadow: 0 6px 24px rgba(139, 92, 246, 0.4);
    }

    .letter-box {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        color: white;
        font-size: 5rem;
        font-weight: 800;
        padding: 2rem;
        border-radius: 20px;
        text-align: center;
        box-shadow: 0 8px 32px rgba(16, 185, 129, 0.4);
    }

    .status-text {
        font-size: 1.2rem;
        font-weight: 600;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)


def load_model():
    """Load sklearn model."""
    model_path = Path("models/asl_model.pickle")
    if not model_path.exists():
        return None, None, None
    try:
        with open(model_path, 'rb') as f:
            data = pickle.load(f)
        return data['model'], data.get('label_encoder'), data.get('scaler')
    except Exception:
        return None, None, None


@st.cache_resource
def get_model():
    return load_model()


def init_state():
    """Initialize session state."""
    if 'current_word' not in st.session_state:
        st.session_state.current_word = ""
    if 'last_spoken_letter' not in st.session_state:
        st.session_state.last_spoken_letter = None
    if 'last_correction_time' not in st.session_state:
        st.session_state.last_correction_time = 0


class ASLProcessor(VideoProcessorBase):
    """Real-time ASL video processor."""

    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )

        model_data = get_model()
        self.model = model_data[0]
        self.label_encoder = model_data[1]
        self.scaler = model_data[2]

        self.lock = threading.Lock()
        self.current_letter = None
        self.confidence = 0.0
        self.hand_detected = False
        self.letter_history = deque(maxlen=10)
        self.stable_letter = None
        self.stability_count = 0
        self.letter_queue = deque(maxlen=5)
        self.last_queued = None
        self.last_queue_time = 0

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.hands.process(img_rgb)

        with self.lock:
            if results.multi_hand_landmarks:
                self.hand_detected = True
                hand = results.multi_hand_landmarks[0]

                # Draw landmarks
                self.mp_drawing.draw_landmarks(
                    img, hand, self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    self.mp_drawing_styles.get_default_hand_connections_style()
                )

                if self.model:
                    # Extract features
                    features = []
                    for lm in hand.landmark:
                        features.extend([lm.x, lm.y])
                    features = np.array(features).reshape(1, -1)

                    if self.scaler:
                        features = self.scaler.transform(features)

                    # Predict
                    pred_idx = self.model.predict(features)[0]

                    if hasattr(self.model, 'predict_proba'):
                        proba = self.model.predict_proba(features)[0]
                        conf = float(np.max(proba))
                    else:
                        conf = 1.0

                    letter = self.label_encoder.inverse_transform([pred_idx])[0]
                    self.current_letter = letter
                    self.confidence = conf

                    if conf >= 0.7:
                        self._update_stability(letter, conf)
                    else:
                        self.stability_count = 0
                        self.stable_letter = None

                    self._draw_overlay(img, letter, conf)
            else:
                self.hand_detected = False
                self.current_letter = None
                self.confidence = 0.0
                self.stability_count = 0
                self.stable_letter = None
                self.letter_history.clear()

        return av.VideoFrame.from_ndarray(img, format="bgr24")

    def _update_stability(self, letter, conf):
        self.letter_history.append(letter)
        if len(self.letter_history) >= 3:
            recent = list(self.letter_history)[-3:]
            if all(l == letter for l in recent):
                self.stability_count += 1
                self.stable_letter = letter

                # Auto-queue after 4 frames
                current_time = time.time()
                if (self.stability_count >= 4 and
                    letter != 'nothing' and
                    letter != self.last_queued and
                    current_time - self.last_queue_time > 0.7):

                    self.letter_queue.append(letter)
                    self.last_queued = letter
                    self.last_queue_time = current_time
            else:
                self.stability_count = max(0, self.stability_count - 1)

    def _draw_overlay(self, img, letter, conf):
        h, w = img.shape[:2]

        # Dark overlay at top
        overlay = img.copy()
        cv2.rectangle(overlay, (0, 0), (w, 90), (20, 30, 48), -1)
        cv2.addWeighted(overlay, 0.85, img, 0.15, 0, img)

        # Letter and confidence
        if self.stable_letter and self.stable_letter != 'nothing':
            color = (100, 255, 100)  # Green
            text = self.stable_letter.upper()
            status = "✓ CONFIRMED"
        else:
            color = (255, 180, 0)  # Orange
            text = letter.upper() if letter != 'nothing' else '—'
            status = "detecting..."

        cv2.putText(img, text, (25, 65), cv2.FONT_HERSHEY_SIMPLEX, 2.2, color, 3)
        cv2.putText(img, f"{conf:.0%}", (140, 65), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        cv2.putText(img, status, (w - 180, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        # Progress bar
        bar_w = int((self.stability_count / 6) * (w - 50))
        cv2.rectangle(img, (25, 78), (25 + bar_w, 85), color, -1)

    def get_queued(self):
        with self.lock:
            if self.letter_queue:
                return self.letter_queue.popleft()
            return None

    def get_state(self):
        with self.lock:
            return {
                'letter': self.stable_letter,
                'confidence': self.confidence,
                'hand_detected': self.hand_detected
            }


def add_letter(letter):
    """Add letter to word and speak it."""
    if letter == 'del':
        if st.session_state.current_word:
            st.session_state.current_word = st.session_state.current_word[:-1]
    elif letter == 'space':
        if st.session_state.current_word and not st.session_state.current_word.endswith(' '):
            st.session_state.current_word += ' '
    elif letter != 'nothing':
        st.session_state.current_word += letter.upper()

        # Auto-speak the letter
        if letter != st.session_state.last_spoken_letter:
            speak_letter(letter)
            st.session_state.last_spoken_letter = letter


def get_correction(word):
    """Get OpenAI correction."""
    if len(word) < 3:
        return None

    # Rate limit
    current_time = time.time()
    if current_time - st.session_state.last_correction_time < 1.5:
        return None

    corrector = get_word_corrector(OPENAI_API_KEY)
    if not corrector.is_available:
        return None

    st.session_state.last_correction_time = current_time
    corrected, _ = corrector.correct_word(word.lower())
    return corrected


def main():
    init_state()

    # Header
    st.markdown("""
    <div class="header-box">
        <h1>🤟 ASL Finger Spelling Recognition</h1>
        <p>Real-time sign language to speech with AI word correction</p>
    </div>
    """, unsafe_allow_html=True)

    # Main layout
    col_video, col_status = st.columns([3, 2])

    with col_video:
        rtc_config = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})

        ctx = webrtc_streamer(
            key="asl",
            mode=WebRtcMode.SENDRECV,
            video_processor_factory=ASLProcessor,
            rtc_configuration=rtc_config,
            media_stream_constraints={"video": {"width": 640, "height": 480}, "audio": False},
            async_processing=True,
        )

    with col_status:
        st.markdown("### Current Letter")

        if ctx.video_processor:
            state = ctx.video_processor.get_state()

            if state['hand_detected'] and state['letter'] and state['letter'] != 'nothing':
                st.markdown(f'<div class="letter-box">{state["letter"].upper()}</div>', unsafe_allow_html=True)
                st.markdown(f'<p class="status-text" style="color: #10b981;">Confidence: {state["confidence"]:.0%}</p>', unsafe_allow_html=True)
            elif state['hand_detected']:
                st.markdown('<p class="status-text" style="color: #f59e0b;">Detecting...</p>', unsafe_allow_html=True)
            else:
                st.markdown('<p class="status-text" style="color: #6b7280;">Show your hand</p>', unsafe_allow_html=True)
        else:
            st.info("Click START to begin")

    # Process auto-queued letters
    if ctx and ctx.video_processor:
        letter = ctx.video_processor.get_queued()
        if letter:
            add_letter(letter)
            st.rerun()

    # Word display
    st.markdown("### Your Word")
    word = st.session_state.current_word if st.session_state.current_word else "..."
    st.markdown(f'<div class="word-display">{word}</div>', unsafe_allow_html=True)

    # AI Correction
    if st.session_state.current_word:
        current_word = st.session_state.current_word.strip().split()[-1] if st.session_state.current_word.strip() else ""
        if len(current_word) >= 3:
            corrected = get_correction(current_word)
            if corrected and corrected.upper() != current_word.upper():
                st.markdown(f'<div class="correction-box">💡 Did you mean: {corrected.upper()}?</div>', unsafe_allow_html=True)

                col1, col2 = st.columns(2)
                with col1:
                    if st.button(f"✓ Use '{corrected.upper()}'", use_container_width=True, type="primary"):
                        words = st.session_state.current_word.strip().split()
                        if words:
                            words[-1] = corrected.upper()
                            st.session_state.current_word = ' '.join(words)
                            speak_word(corrected)
                            st.rerun()
                with col2:
                    if st.button("✗ Ignore", use_container_width=True):
                        st.session_state.last_correction_time = time.time() + 10  # Delay next correction
                        st.rerun()

    # Controls
    st.markdown("---")
    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("🔊 Speak Word", use_container_width=True, type="primary"):
            if st.session_state.current_word:
                speak_word(st.session_state.current_word)

    with col2:
        if st.button("␣ Add Space", use_container_width=True):
            if st.session_state.current_word and not st.session_state.current_word.endswith(' '):
                st.session_state.current_word += ' '
                st.rerun()

    with col3:
        if st.button("🗑️ Clear All", use_container_width=True):
            st.session_state.current_word = ""
            st.session_state.last_spoken_letter = None
            st.rerun()

    # Model status
    st.markdown("---")
    model, _, _ = get_model()
    if model:
        st.success(f"✓ Model ready: {model.__class__.__name__}")
    else:
        st.error("✗ No model found - train the model first")


if __name__ == "__main__":
    main()
