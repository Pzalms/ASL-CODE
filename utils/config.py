"""
Configuration for ASL Finger Spelling Recognition System
"""

from pathlib import Path
from utils.features import TOTAL_FEATURES

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "asl_model.pickle"
COLLECTED_DATA_DIR = PROJECT_ROOT / "collected_data"

# Model configuration
NUM_CLASSES = 28  # A-Z + del + space (no 'nothing' — handled by hand detection)
FEATURE_DIM = TOTAL_FEATURES  # 80

# Class labels
CLASS_LABELS = [
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
    'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
    'del', 'space',
]

# Recognition settings
CONFIDENCE_THRESHOLD = 0.65
STABILITY_FRAMES = 5

# UI configuration
PAGE_TITLE = "ASL Finger Spelling"
PAGE_ICON = "🤟"

# TTS configuration
TTS_RATE = 160
TTS_VOLUME = 1.0
