"""
Configuration for ASL Finger Spelling Recognition System
"""

from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "asl_landmark_model.keras"
COLLECTED_DATA_DIR = PROJECT_ROOT / "collected_data"

# Model configuration
NUM_CLASSES = 29
FEATURE_DIM = 73  # 21 landmarks * 3 coords + 10 angles

# Class labels
CLASS_LABELS = [
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
    'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
    'del', 'nothing', 'space'
]

# Recognition settings
CONFIDENCE_THRESHOLD = 0.70
STABILITY_FRAMES = 4  # Frames to hold for letter confirmation

# UI configuration
PAGE_TITLE = "ASL Finger Spelling"
PAGE_ICON = "🤟"

# TTS configuration
TTS_RATE = 160
TTS_VOLUME = 1.0
