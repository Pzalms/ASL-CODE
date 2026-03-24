#!/usr/bin/env python3
"""
Quick camera test — verify hand sign detection works in real-time.
Opens a window showing the webcam with live predictions.

Usage:  python3 test_camera.py
Keys:   q = quit
"""

import cv2
import numpy as np
import pickle
import time
from pathlib import Path
from collections import deque

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

from utils.features import extract_features

HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
    (5,9),(9,13),(13,17),
]


def main():
    # Load model
    with open("models/asl_model.pickle", "rb") as f:
        data = pickle.load(f)
    model, le = data["model"], data["label_encoder"]
    scaler = data.get("scaler")
    print(f"Model loaded: {type(model).__name__}, {len(le.classes_)} classes, {data['feature_dim']}D")

    # Create hand detector
    base = mp_python.BaseOptions(
        model_asset_path=str(Path("models/hand_landmarker.task").resolve())
    )
    detector = mp_vision.HandLandmarker.create_from_options(
        mp_vision.HandLandmarkerOptions(
            base_options=base,
            running_mode=mp_vision.RunningMode.IMAGE,
            num_hands=1,
            min_hand_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
    )

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        print("ERROR: Cannot open camera")
        return

    history = deque(maxlen=10)
    stable_letter = None
    word = ""
    fps_times = deque(maxlen=30)

    print("Camera open. Show ASL hand signs. Press 'q' to quit.")

    while True:
        t0 = time.time()
        ret, frame = cap.read()
        if not ret:
            break

        # No flip — training data is from camera perspective, not mirrored
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = detector.detect(mp_image)

        letter = None
        conf = 0.0

        if result.hand_landmarks:
            hand_lms = result.hand_landmarks[0]

            # Draw hand
            pts = [(int(lm.x * w), int(lm.y * h)) for lm in hand_lms]
            for i, j in HAND_CONNECTIONS:
                cv2.line(frame, pts[i], pts[j], (0, 220, 0), 2, cv2.LINE_AA)
            for x, y in pts:
                cv2.circle(frame, (x, y), 4, (255, 50, 50), -1, cv2.LINE_AA)

            # Predict
            landmarks_np = np.array([[lm.x, lm.y, lm.z] for lm in hand_lms])
            feats = extract_features(landmarks_np)
            if feats is not None:
                f2d = feats.reshape(1, -1)
                if scaler:
                    f2d = scaler.transform(f2d)
                proba = model.predict_proba(f2d)[0]
                pred_idx = int(np.argmax(proba))
                conf = float(proba[pred_idx])
                letter = le.inverse_transform([pred_idx])[0]

                # Stability
                if conf >= 0.6:
                    history.append(letter)
                    recent = list(history)[-5:]
                    if len(recent) == 5 and all(r == letter for r in recent):
                        stable_letter = letter
                    else:
                        stable_letter = None
                else:
                    history.clear()
                    stable_letter = None

                # Show top 3 predictions
                top3_idx = np.argsort(proba)[-3:][::-1]
                for rank, idx in enumerate(top3_idx):
                    lbl = le.inverse_transform([idx])[0]
                    p = proba[idx]
                    color = (0, 255, 0) if rank == 0 else (180, 180, 180)
                    cv2.putText(frame, f"{lbl}: {p:.0%}", (w - 150, 30 + rank * 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        else:
            history.clear()
            stable_letter = None

        # ─── HUD ──────────────────────────────────────────────────────────
        # Top bar
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 70), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        if stable_letter and stable_letter != "nothing":
            cv2.putText(frame, stable_letter.upper(), (20, 52),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.8, (0, 255, 100), 3)
            cv2.putText(frame, "STABLE", (100, 52),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 100), 2)
        elif letter:
            col = (0, 200, 255) if conf >= 0.6 else (0, 140, 255)
            cv2.putText(frame, letter.upper(), (20, 52),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.8, col, 3)
            cv2.putText(frame, f"{conf:.0%}", (100, 52),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        else:
            cv2.putText(frame, "No hand", (20, 48),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (100, 100, 100), 2)

        # Bottom bar - word
        overlay2 = frame.copy()
        cv2.rectangle(overlay2, (0, h - 50), (w, h), (0, 0, 0), -1)
        cv2.addWeighted(overlay2, 0.6, frame, 0.4, 0, frame)
        cv2.putText(frame, f"Word: {word}", (20, h - 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # FPS
        fps_times.append(time.time() - t0)
        fps = len(fps_times) / sum(fps_times) if fps_times else 0
        cv2.putText(frame, f"{fps:.0f} FPS", (w - 100, h - 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 1)

        cv2.imshow("ASL Test - press q to quit", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord(" ") and stable_letter and stable_letter != "nothing":
            word += stable_letter.upper()
            print(f"  Added: {stable_letter.upper()}  ->  {word}")
        elif key == 8:  # backspace
            word = word[:-1]
        elif key == 13:  # enter
            print(f"  WORD: {word}")
            word = ""

    cap.release()
    cv2.destroyAllWindows()
    detector.close()
    print("Done.")


if __name__ == "__main__":
    main()
