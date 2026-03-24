#!/usr/bin/env python3
"""
ASL Recognition – Model Training Script

Extracts hand landmarks via MediaPipe Tasks API, computes normalised
features (utils/features.py), augments with flips + noise, and trains
a HistGradientBoosting classifier.

Usage:
    python3 train.py
    python3 train.py --use-cache   # skip extraction, retrain only
"""

import argparse
import pickle
import time
import sys
import numpy as np
import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from pathlib import Path
from collections import Counter
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

from utils.features import extract_features, TOTAL_FEATURES, N_POSITION

# ─── Paths ────────────────────────────────────────────────────────────────────
DATASET_DIR = Path('dataset/asl_alphabet_train/asl_alphabet_train')
MODEL_DIR = Path('models')
MODEL_DIR.mkdir(exist_ok=True)
MODEL_PATH = MODEL_DIR / 'asl_model.pickle'
DATA_CACHE = MODEL_DIR / 'landmark_data.pickle'

LABELS = [
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
    'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
    'del', 'space',
]

# Classes that are easily confused — process more images for these
HARD_CLASSES = {'M', 'N', 'T', 'U', 'E', 'S', 'A', 'R'}


# ─── Augmentation ─────────────────────────────────────────────────────────────

def flip_features(features):
    """Horizontally flip: negate x-coords in position block."""
    f = features.copy()
    for i in range(0, N_POSITION, 3):
        f[i] = -f[i]
    return f


def add_noise(features, scale=0.015):
    return features + np.random.normal(0, scale, features.shape)


def augment(X, y):
    """4× augmentation: original + flipped + 2× noisy."""
    X_flip = np.array([flip_features(f) for f in X])
    return (
        np.vstack([X, X_flip, add_noise(X, 0.015), add_noise(X_flip, 0.015)]),
        np.concatenate([y, y, y, y]),
    )


# ─── Feature extraction ──────────────────────────────────────────────────────

def extract_dataset() -> tuple:
    model_path = str((Path('models') / 'hand_landmarker.task').resolve())
    detector = mp_vision.HandLandmarker.create_from_options(
        mp_vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=model_path),
            running_mode=mp_vision.RunningMode.IMAGE,
            num_hands=1,
            min_hand_detection_confidence=0.5,
        )
    )

    all_features, all_labels = [], []

    for label in tqdm(LABELS, desc='Extracting', unit='class'):
        label_dir = DATASET_DIR / label
        if not label_dir.exists():
            tqdm.write(f'  ⚠ {label_dir} not found')
            continue

        # More images for confusing classes
        limit = 3000 if label in HARD_CLASSES else 1000

        images = sorted(label_dir.glob('*.jpg')) + sorted(label_dir.glob('*.png'))
        images = images[:limit]

        ok = 0
        for img_path in images:
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            result = detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))

            if result.hand_landmarks:
                lms = np.array([[l.x, l.y, l.z] for l in result.hand_landmarks[0]])
                feats = extract_features(lms)
                if feats is not None:
                    all_features.append(feats)
                    all_labels.append(label)
                    ok += 1

        tqdm.write(f'  {label}: {ok}/{len(images)} detected')

    detector.close()
    return np.array(all_features), np.array(all_labels)


# ─── Training ────────────────────────────────────────────────────────────────

def train(X, y):
    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=0.15, random_state=42, stratify=y_enc,
    )

    print(f'\nTrain: {len(X_train)}  |  Test: {len(X_test)}')

    # Scale features for MLP
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    print('Training MLP …')
    model = MLPClassifier(
        hidden_layer_sizes=(512, 256, 128),
        activation='relu',
        max_iter=300,
        early_stopping=True,
        validation_fraction=0.1,
        batch_size=256,
        learning_rate='adaptive',
        learning_rate_init=0.001,
        random_state=42,
        verbose=True,
    )
    model.fit(X_train_s, y_train)

    y_pred = model.predict(X_test_s)
    acc = accuracy_score(y_test, y_pred)

    print(f'\n{"=" * 60}')
    print(f'  Accuracy: {acc:.2%}')
    print(f'{"=" * 60}')
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    return model, le, acc, scaler


# ─── Save ─────────────────────────────────────────────────────────────────────

def save(model, le, acc, scaler, X_raw, y_raw):
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump({
            'model': model,
            'label_encoder': le,
            'scaler': scaler,
            'accuracy': acc,
            'feature_dim': TOTAL_FEATURES,
            'n_samples': len(X_raw),
            'class_names': list(le.classes_),
        }, f)
    print(f'Model saved → {MODEL_PATH}')

    with open(DATA_CACHE, 'wb') as f:
        pickle.dump({'data': X_raw.tolist(), 'labels': y_raw.tolist()}, f)
    print(f'Data cache saved → {DATA_CACHE}')


# ─── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Train ASL recognition model')
    parser.add_argument('--use-cache', action='store_true',
                        help='Skip extraction, retrain from cached data')
    args = parser.parse_args()

    print('ASL Model Training')
    print(f'{"=" * 60}')
    print(f'Feature dimension : {TOTAL_FEATURES}')

    if not DATASET_DIR.exists() and not args.use_cache:
        sys.exit(f'ERROR: dataset not found at {DATASET_DIR}')

    # ── Extract or load cache ─────────────────────────────────────────────
    need_extract = True
    if args.use_cache and DATA_CACHE.exists():
        print('Loading cached feature data …')
        with open(DATA_CACHE, 'rb') as f:
            cache = pickle.load(f)
        X_raw = np.array(cache['data'])
        y_raw = np.array(cache['labels'])
        if X_raw.shape[1] == TOTAL_FEATURES:
            need_extract = False
        else:
            print(f'Cache has {X_raw.shape[1]}D, need {TOTAL_FEATURES}D — re-extracting')

    if need_extract:
        t0 = time.time()
        X_raw, y_raw = extract_dataset()
        print(f'\nExtraction: {len(X_raw)} samples in {(time.time()-t0)/60:.1f} min')

    print(f'Raw data: {X_raw.shape}')
    counts = Counter(y_raw)
    print('\nSamples per class:')
    for lbl in sorted(counts):
        print(f'  {lbl}: {counts[lbl]}')

    # ── Augment ────────────────────────────────────────────────────────────
    print('\nAugmenting (flip + noise) …')
    X_aug, y_aug = augment(X_raw, y_raw)
    print(f'Augmented: {X_aug.shape} ({len(X_aug)/len(X_raw):.0f}×)')

    # ── Train ──────────────────────────────────────────────────────────────
    model, le, acc, scaler = train(X_aug, y_aug)

    # ── Save ──────────────────────────────────────────────────────────────
    save(model, le, acc, scaler, X_raw, y_raw)
    print(f'\nDone — accuracy {acc:.2%}')


if __name__ == '__main__':
    main()
