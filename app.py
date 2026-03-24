"""
ASL Finger Spelling Recognition
Real-time hand sign → text → speech, with predictive word intelligence.

Two modes:
  Letter Mode — raw letters, no AI (for testing individual signs)
  Word Mode   — predictive texting: each letter triggers word suggestions
"""

import os
from dotenv import load_dotenv
load_dotenv()

import time
import json
import cv2
import numpy as np
import pickle
import streamlit as st
from streamlit_webrtc import (
    webrtc_streamer, VideoProcessorBase, RTCConfiguration, WebRtcMode,
)
import av
import threading
from collections import deque
from pathlib import Path

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

from utils.features import extract_features
from utils.tts_handler import speak_letter, speak_word

# ─── Hand drawing ────────────────────────────────────────────────────────────

HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),(0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),(5,9),(9,13),(13,17),
]

def draw_hand(img, lms):
    h, w = img.shape[:2]
    pts = [(int(l.x * w), int(l.y * h)) for l in lms]
    for i, j in HAND_CONNECTIONS:
        cv2.line(img, pts[i], pts[j], (0, 230, 0), 2, cv2.LINE_AA)
    for x, y in pts:
        cv2.circle(img, (x, y), 4, (255, 50, 50), -1, cv2.LINE_AA)


# ─── Page config ─────────────────────────────────────────────────────────────

st.set_page_config(page_title="ASL Recognition", page_icon="🤟", layout="wide",
                   initial_sidebar_state="collapsed")

st.markdown("""
<style>
#MainMenu,footer,header{visibility:hidden}
.stDeployButton{display:none}
.main .block-container{padding:1.5rem 2rem;max-width:1200px}

.app-title{display:flex;align-items:center;gap:.75rem;padding:1rem 1.5rem;
  background:#111827;border-radius:12px;margin-bottom:1.5rem;border:1px solid #1f2937}
.app-title h1{color:#f9fafb;font-size:1.5rem;font-weight:700;margin:0}
.app-title p{color:#6b7280;font-size:.85rem;margin:0}

.det-letter{background:#111827;border:2px solid #1f2937;border-radius:16px;
  padding:1.5rem;text-align:center;margin-bottom:1rem}
.det-letter .lbl{color:#6b7280;font-size:.75rem;text-transform:uppercase;letter-spacing:.1em;margin-bottom:.5rem}
.det-letter .big{font-size:5rem;font-weight:800;line-height:1;margin:.25rem 0}
.det-letter .big.ok{color:#10b981} .det-letter .big.wait{color:#f59e0b} .det-letter .big.off{color:#374151}

.conf-bar{width:100%;height:6px;background:#1f2937;border-radius:3px;margin-top:.75rem;overflow:hidden}
.conf-fill{height:100%;border-radius:3px}
.conf-fill.hi{background:#10b981}.conf-fill.md{background:#f59e0b}.conf-fill.lo{background:#ef4444}

.meta{display:flex;justify-content:space-between;margin-top:.5rem}
.meta span{color:#6b7280;font-size:.8rem}

.wcard{background:#111827;border:1px solid #1f2937;border-radius:12px;padding:1.25rem 1.5rem;margin-bottom:1rem}
.wcard .lbl{color:#6b7280;font-size:.75rem;text-transform:uppercase;letter-spacing:.1em;margin-bottom:.5rem}
.wcard .txt{color:#f9fafb;font-size:1.6rem;font-weight:700;letter-spacing:.12em;
  font-family:'SF Mono','Fira Code',monospace;min-height:2rem;word-break:break-all}
.wcard .txt.empty{color:#374151}

/* Sentence display */
.sentence{background:#0f172a;border:1px solid #1e293b;border-radius:12px;padding:1rem 1.5rem;margin-bottom:1rem}
.sentence .lbl{color:#6b7280;font-size:.75rem;text-transform:uppercase;letter-spacing:.1em;margin-bottom:.4rem}
.sentence .txt{color:#e2e8f0;font-size:1.3rem;font-weight:600;min-height:1.5rem;line-height:1.6}

/* Prediction chips */
.pred-bar{display:flex;gap:.5rem;flex-wrap:wrap;margin-bottom:1rem}
.pred-chip{background:#1e1b4b;border:1px solid #312e81;border-radius:8px;padding:.5rem 1rem;
  color:#c4b5fd;font-weight:700;font-size:1rem;cursor:pointer;letter-spacing:.05em;
  transition:all .15s ease}
.pred-chip:hover{background:#312e81;color:#e0e7ff}
.pred-chip.top{background:#2563eb;border-color:#3b82f6;color:#fff;font-size:1.1rem}

/* Mode toggle */
.mode-toggle{display:flex;gap:0;margin-bottom:1rem;border-radius:8px;overflow:hidden;border:1px solid #374151}
.mode-btn{flex:1;padding:.6rem;text-align:center;font-weight:700;font-size:.85rem;
  cursor:pointer;transition:all .15s ease;color:#9ca3af;background:#111827}
.mode-btn.active{background:#2563eb;color:#fff}

.badge{display:inline-flex;align-items:center;gap:.4rem;padding:.35rem .75rem;
  border-radius:20px;font-size:.75rem;font-weight:600}
.badge.ok{background:#052e16;color:#4ade80;border:1px solid #166534}
.badge.err{background:#450a0a;color:#f87171;border:1px solid #991b1b}
.badge .dot{width:6px;height:6px;border-radius:50%;display:inline-block}
.badge.ok .dot{background:#4ade80}.badge.err .dot{background:#f87171}

.stButton>button{background:#1f2937!important;color:#f9fafb!important;border:1px solid #374151!important;
  border-radius:8px!important;font-weight:600!important;padding:.5rem 1rem!important}
.stButton>button:hover{background:#374151!important;border-color:#4b5563!important}
.stButton>button[kind="primary"]{background:#2563eb!important;border-color:#2563eb!important}
.stButton>button[kind="primary"]:hover{background:#1d4ed8!important}
</style>
""", unsafe_allow_html=True)


# ─── Model ───────────────────────────────────────────────────────────────────

@st.cache_resource
def load_model():
    p = Path("models/asl_model.pickle")
    if not p.exists():
        return None, None, None
    with open(p, "rb") as f:
        d = pickle.load(f)
    return d["model"], d["label_encoder"], d.get("scaler")


# ─── OpenAI predictive engine ────────────────────────────────────────────────

@st.cache_resource
def get_openai_client():
    key = os.environ.get("OPENAI_API_KEY", "")
    if not key:
        return None
    try:
        from openai import OpenAI
        return OpenAI(api_key=key)
    except Exception:
        return None


def ai_predict_words(letters: str, sentence_so_far: str = "") -> list:
    """Predictive texting: given letters so far, return list of likely words.

    Returns up to 5 word suggestions, best first.
    """
    if not letters:
        return []

    # Rate limit — 1 call per second
    now = time.time()
    last = st.session_state.get("_ai_last_call", 0)
    last_letters = st.session_state.get("_ai_last_letters", "")

    # Return cache if same letters and recent call
    if letters == last_letters and now - last < 1.0:
        return st.session_state.get("_ai_predictions", [])

    client = get_openai_client()
    if not client:
        return []

    st.session_state["_ai_last_call"] = now
    st.session_state["_ai_last_letters"] = letters

    ctx = f'\nSentence so far: "{sentence_so_far}"' if sentence_so_far else ""

    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": """You are a predictive text engine for ASL finger spelling recognition.

Given letters spelled so far (which may contain recognition errors), suggest the 5 most likely English words the person is trying to spell.

Common ASL recognition confusions: M↔N, A↔S↔E↔T, U↔V↔R, D↔F, K↔V, I↔J

Return ONLY a JSON array of 5 words, most likely first:
["word1", "word2", "word3", "word4", "word5"]

Rules:
- All suggestions must be real, common English words
- Consider the sentence context to rank suggestions
- Consider which ASL letters are commonly confused
- If only 1 letter, suggest common words starting with that letter
- Words should make sense in the sentence context
- Return shorter/common words first when letters are few"""},
                {"role": "user", "content": f"Letters: {letters}{ctx}"}
            ],
            temperature=0.3,
            max_tokens=60,
        )

        result = json.loads(resp.choices[0].message.content.strip())
        if isinstance(result, list):
            words = [w.upper() for w in result if isinstance(w, str) and w.strip()][:5]
            st.session_state["_ai_predictions"] = words
            return words
    except Exception:
        pass

    return st.session_state.get("_ai_predictions", [])


# ─── Session state ───────────────────────────────────────────────────────────

def init_state():
    for k, v in {
        "current_word": "",
        "full_text": "",
        "last_spoken": None,
        "tts_enabled": True,
        "mode": "word",           # "word" or "letter"
        "_ai_last_call": 0.0,
        "_ai_last_letters": "",
        "_ai_predictions": [],
    }.items():
        if k not in st.session_state:
            st.session_state[k] = v


# ─── Video Processor ─────────────────────────────────────────────────────────

class ASLProcessor(VideoProcessorBase):

    STABILITY_REQUIRED = 5
    COOLDOWN = 0.8
    CONF_THRESHOLD = 0.55

    def __init__(self):
        base = mp_python.BaseOptions(
            model_asset_path=str(Path("models/hand_landmarker.task").resolve())
        )
        self.detector = mp_vision.HandLandmarker.create_from_options(
            mp_vision.HandLandmarkerOptions(
                base_options=base,
                running_mode=mp_vision.RunningMode.IMAGE,
                num_hands=1,
                min_hand_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
        )

        model, le, scaler = load_model()
        self.model = model
        self.le = le
        self.scaler = scaler

        self.lock = threading.Lock()
        self.letter = None
        self.conf = 0.0
        self.hand_ok = False
        self.history = deque(maxlen=15)
        self.stable = None
        self.stab_count = 0
        self.queue = deque(maxlen=5)
        self.last_q = None
        self.last_q_time = 0.0

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self.detector.detect(mp_img)

        with self.lock:
            if result.hand_landmarks:
                lms = result.hand_landmarks[0]
                self.hand_ok = True
                draw_hand(img, lms)
                if self.model:
                    arr = np.array([[l.x, l.y, l.z] for l in lms])
                    self._predict(arr)
                    self._overlay(img)
            else:
                self._clear()

        return av.VideoFrame.from_ndarray(img, format="bgr24")

    def _predict(self, landmarks):
        feats = extract_features(landmarks)
        if feats is None:
            self._clear()
            return

        f2d = feats.reshape(1, -1)
        if self.scaler:
            f2d = self.scaler.transform(f2d)
        proba = self.model.predict_proba(f2d)[0]
        idx = int(np.argmax(proba))
        c = float(proba[idx])
        ltr = self.le.inverse_transform([idx])[0]

        self.letter = ltr
        self.conf = c

        if c < self.CONF_THRESHOLD:
            self.stab_count = max(0, self.stab_count - 1)
            return

        self.history.append(ltr)
        recent = list(self.history)[-self.STABILITY_REQUIRED:]
        if len(recent) == self.STABILITY_REQUIRED and all(r == ltr for r in recent):
            self.stable = ltr
            self.stab_count = self.STABILITY_REQUIRED
            now = time.time()
            if ltr != "nothing" and (ltr != self.last_q or now - self.last_q_time > self.COOLDOWN):
                self.queue.append(ltr)
                self.last_q = ltr
                self.last_q_time = now
        else:
            cnt = 0
            for r in reversed(recent):
                if r == ltr:
                    cnt += 1
                else:
                    break
            self.stab_count = cnt
            if cnt < 2:
                self.stable = None

    def _clear(self):
        self.hand_ok = False
        self.letter = None
        self.conf = 0.0
        self.stab_count = 0
        self.stable = None
        self.history.clear()

    def _overlay(self, img):
        h, w = img.shape[:2]
        ov = img.copy()
        cv2.rectangle(ov, (0, 0), (w, 70), (0, 0, 0), -1)
        cv2.addWeighted(ov, 0.65, img, 0.35, 0, img)
        ltr = self.letter or ""
        c = self.conf
        if self.stable and self.stable != "nothing":
            color = (100, 255, 100)
            txt, tag = self.stable.upper(), "CONFIRMED"
        else:
            color = (0, 200, 255)
            txt = ltr.upper() if ltr and ltr != "nothing" else ""
            tag = f"{c:.0%}" if ltr else ""
        cv2.putText(img, txt, (20, 52), cv2.FONT_HERSHEY_SIMPLEX, 1.8, color, 3)
        cv2.putText(img, tag, (w - 170, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        pct = self.stab_count / self.STABILITY_REQUIRED
        cv2.rectangle(img, (20, 60), (20 + int(pct * (w - 40)), 65), color, -1)

    def pop_letter(self):
        with self.lock:
            return self.queue.popleft() if self.queue else None

    def state(self):
        with self.lock:
            return dict(letter=self.stable, raw=self.letter, conf=self.conf,
                        hand=self.hand_ok, stab=self.stab_count)


# ─── Helpers ─────────────────────────────────────────────────────────────────

def add_letter(ltr):
    if ltr == "del":
        if st.session_state.current_word:
            st.session_state.current_word = st.session_state.current_word[:-1]
    elif ltr == "space":
        w = st.session_state.current_word.strip()
        if w:
            st.session_state.full_text += w + " "
            st.session_state.current_word = ""
    elif ltr != "nothing":
        st.session_state.current_word += ltr.upper()
        if ltr != st.session_state.last_spoken:
            speak_letter(ltr)
            st.session_state.last_spoken = ltr
    # Reset predictions on change
    st.session_state["_ai_predictions"] = []
    st.session_state["_ai_last_letters"] = ""


def accept_word(word):
    """Accept a predicted word — add to sentence, clear current letters."""
    st.session_state.full_text += word.upper() + " "
    st.session_state.current_word = ""
    st.session_state["_ai_predictions"] = []
    st.session_state["_ai_last_letters"] = ""
    speak_word(word.lower())


def get_sentence():
    return st.session_state.full_text.strip()


def get_full_display():
    t = st.session_state.full_text + st.session_state.current_word
    return t.strip()


# ─── UI ──────────────────────────────────────────────────────────────────────

def main():
    init_state()

    model, le, _ = load_model()
    badge = '<span class="badge ok"><span class="dot"></span>Model ready</span>' if model \
        else '<span class="badge err"><span class="dot"></span>No model</span>'

    ai_ok = get_openai_client() is not None
    ai_badge = '<span class="badge ok"><span class="dot"></span>AI active</span>' if ai_ok \
        else '<span class="badge err"><span class="dot"></span>AI off</span>'

    st.markdown(f"""
    <div class="app-title"><div>
        <h1>🤟 ASL Sign Language Recognition</h1>
        <p>Hand sign detection · Finger spelling · Predictive text &nbsp; {badge} &nbsp; {ai_badge}</p>
    </div></div>""", unsafe_allow_html=True)

    # ── Layout ────────────────────────────────────────────────────────────
    col_cam, col_info = st.columns([3, 2], gap="large")

    with col_cam:
        ctx = webrtc_streamer(
            key="asl", mode=WebRtcMode.SENDRECV,
            video_processor_factory=ASLProcessor,
            rtc_configuration=RTCConfiguration(
                {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}),
            media_stream_constraints={"video": {"width": 640, "height": 480}, "audio": False},
            async_processing=True,
        )

    with col_info:
        # ── Mode toggle ───────────────────────────────────────────────────
        m1, m2 = st.columns(2)
        with m1:
            if st.button("📝 Word Mode" + (" ●" if st.session_state.mode == "word" else ""),
                         use_container_width=True,
                         type="primary" if st.session_state.mode == "word" else "secondary"):
                st.session_state.mode = "word"
                st.rerun()
        with m2:
            if st.button("🔤 Letter Mode" + (" ●" if st.session_state.mode == "letter" else ""),
                         use_container_width=True,
                         type="primary" if st.session_state.mode == "letter" else "secondary"):
                st.session_state.mode = "letter"
                st.rerun()

        # ── Detected sign ─────────────────────────────────────────────────
        if ctx.video_processor:
            s = ctx.video_processor.state()
            if s["hand"] and s["letter"] and s["letter"] != "nothing":
                ld, lc = s["letter"].upper(), "ok"
                cp = int(s["conf"] * 100)
                cc = "hi" if cp >= 80 else "md" if cp >= 60 else "lo"
                sp = int((s["stab"] / ASLProcessor.STABILITY_REQUIRED) * 100)
            elif s["hand"]:
                raw = s.get("raw", "")
                ld = raw.upper() if raw and raw != "nothing" else "..."
                lc, cp, cc = "wait", int(s["conf"] * 100), "md"
                sp = int((s["stab"] / ASLProcessor.STABILITY_REQUIRED) * 100)
            else:
                ld, lc, cp, cc, sp = "—", "off", 0, "lo", 0
        else:
            ld, lc, cp, cc, sp = "—", "off", 0, "lo", 0

        st.markdown(f"""
        <div class="det-letter">
            <div class="lbl">Detected Sign</div>
            <div class="big {lc}">{ld}</div>
            <div class="conf-bar"><div class="conf-fill {cc}" style="width:{cp}%"></div></div>
            <div class="meta"><span>Confidence: {cp}%</span><span>Stability: {sp}%</span></div>
        </div>""", unsafe_allow_html=True)

        # ── Current letters being spelled ─────────────────────────────────
        cw = st.session_state.current_word.strip()
        mode = st.session_state.mode

        if mode == "word":
            # Show current letters
            st.markdown(f"""
            <div class="wcard">
                <div class="lbl">Spelling</div>
                <div class="txt {"empty" if not cw else ""}">{cw or "Sign a letter..."}</div>
            </div>""", unsafe_allow_html=True)

            # ── Predictive suggestions ────────────────────────────────────
            if ai_ok and cw:
                predictions = ai_predict_words(cw, get_sentence())
                if predictions:
                    # Render prediction buttons
                    cols = st.columns(min(len(predictions), 5))
                    for i, word in enumerate(predictions):
                        with cols[i]:
                            btn_type = "primary" if i == 0 else "secondary"
                            if st.button(word, key=f"pred_{i}", use_container_width=True, type=btn_type):
                                accept_word(word)
                                st.rerun()
            elif not ai_ok and cw:
                st.caption("Set OPENAI_API_KEY for word predictions")

            # ── Sentence so far ───────────────────────────────────────────
            sentence = get_sentence()
            if sentence:
                st.markdown(f"""
                <div class="sentence">
                    <div class="lbl">Sentence</div>
                    <div class="txt">{sentence}</div>
                </div>""", unsafe_allow_html=True)

        else:
            # Letter mode — just raw letter accumulation
            full = get_full_display()
            st.markdown(f"""
            <div class="wcard">
                <div class="lbl">Letters</div>
                <div class="txt {"empty" if not full else ""}">{full or "Sign a letter..."}</div>
            </div>""", unsafe_allow_html=True)

        # ── Controls ──────────────────────────────────────────────────────
        st.markdown("<br>", unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            if st.button("🔊 Speak", use_container_width=True, type="primary"):
                t = get_sentence() if mode == "word" else get_full_display()
                if t:
                    speak_word(t)
        with c2:
            if mode == "word":
                # In word mode, space = confirm current letters as-is (no prediction)
                if st.button("✓ Next Word", use_container_width=True):
                    if cw:
                        accept_word(cw)
                        st.rerun()
            else:
                if st.button("␣ Space", use_container_width=True):
                    add_letter("space")
                    st.rerun()
        with c3:
            if st.button("⌫ Delete", use_container_width=True):
                add_letter("del")
                st.rerun()
        with c4:
            if st.button("🗑 Clear", use_container_width=True):
                st.session_state.current_word = ""
                st.session_state.full_text = ""
                st.session_state.last_spoken = None
                st.session_state["_ai_predictions"] = []
                st.rerun()

    # ── Process queued letters ────────────────────────────────────────────
    if ctx and ctx.video_processor:
        ltr = ctx.video_processor.pop_letter()
        if ltr:
            add_letter(ltr)
            st.rerun()


if __name__ == "__main__":
    main()
