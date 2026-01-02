"""
Data Collection Tool for ASL Finger Spelling

Collects hand landmark data from webcam for training the finger spelling model.
"""

import os
import cv2
import numpy as np
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Optional
from .hand_detector import HandDetector


class DataCollector:
    """
    Collects hand landmark data for ASL training.

    Records landmark features for each letter with timestamp and metadata.
    """

    LABELS = [
        'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
        'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
        'del', 'nothing', 'space'
    ]

    def __init__(self, save_dir: str = "collected_data"):
        """
        Args:
            save_dir: Directory to save collected data
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.detector = HandDetector(max_hands=1)
        self.current_label = None
        self.samples = []

        # Create label directories
        for label in self.LABELS:
            (self.save_dir / label).mkdir(exist_ok=True)

    def set_label(self, label: str):
        """Set current label for data collection."""
        if label in self.LABELS:
            self.current_label = label
            print(f"Now collecting: {label}")
        else:
            print(f"Invalid label: {label}")

    def collect_sample(self, image: np.ndarray) -> Optional[dict]:
        """
        Collect a single sample from image.

        Args:
            image: BGR image from webcam

        Returns:
            Sample dict if hand detected, None otherwise
        """
        if self.current_label is None:
            return None

        hands = self.detector.detect(image)
        if not hands:
            return None

        hand = hands[0]
        features = self.detector.get_combined_features(hand.landmarks)

        sample = {
            'label': self.current_label,
            'features': features.tolist(),
            'landmarks': hand.landmarks.tolist(),
            'handedness': hand.handedness,
            'confidence': hand.confidence,
            'timestamp': datetime.now().isoformat()
        }

        self.samples.append(sample)
        return sample

    def save_samples(self):
        """Save collected samples to disk."""
        if not self.samples:
            print("No samples to save")
            return

        # Group by label
        by_label = {}
        for sample in self.samples:
            label = sample['label']
            if label not in by_label:
                by_label[label] = []
            by_label[label].append(sample)

        # Save each label's samples
        for label, samples in by_label.items():
            filename = f"{label}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            filepath = self.save_dir / label / filename

            with open(filepath, 'w') as f:
                json.dump(samples, f)

            print(f"Saved {len(samples)} samples for '{label}' to {filepath}")

        self.samples = []

    def get_stats(self) -> dict:
        """Get statistics about collected data."""
        stats = {}

        for label in self.LABELS:
            label_dir = self.save_dir / label
            files = list(label_dir.glob("*.json"))
            total_samples = 0

            for f in files:
                with open(f, 'r') as fp:
                    data = json.load(fp)
                    total_samples += len(data)

            stats[label] = total_samples

        return stats

    def close(self):
        """Release resources."""
        self.detector.close()


def load_collected_data(data_dir: str = "collected_data") -> tuple:
    """
    Load all collected data for training.

    Args:
        data_dir: Directory containing collected data

    Returns:
        (features, labels) numpy arrays
    """
    data_path = Path(data_dir)
    all_features = []
    all_labels = []

    label_to_idx = {label: idx for idx, label in enumerate(DataCollector.LABELS)}

    for label in DataCollector.LABELS:
        label_dir = data_path / label
        if not label_dir.exists():
            continue

        for json_file in label_dir.glob("*.json"):
            with open(json_file, 'r') as f:
                samples = json.load(f)

            for sample in samples:
                all_features.append(sample['features'])
                all_labels.append(label_to_idx[sample['label']])

    if not all_features:
        return np.array([]), np.array([])

    return np.array(all_features), np.array(all_labels)


def run_collection_session(samples_per_letter: int = 100, delay: float = 0.1):
    """
    Interactive data collection session.

    Args:
        samples_per_letter: Target samples per letter
        delay: Delay between captures in seconds
    """
    collector = DataCollector()

    print("\n" + "=" * 50)
    print("ASL Data Collection Tool")
    print("=" * 50)
    print("\nControls:")
    print("  Press a letter key (A-Z) to start collecting that letter")
    print("  Press SPACE to collect 'space' gesture")
    print("  Press BACKSPACE to collect 'del' gesture")
    print("  Press 'n' to collect 'nothing' (neutral)")
    print("  Press 's' to save collected data")
    print("  Press 'q' to quit")
    print("\n")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return

    collecting = False
    collect_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Mirror the frame
        frame = cv2.flip(frame, 1)

        # Detect hand and draw
        hands = collector.detector.detect(frame)
        if hands:
            frame = collector.detector.draw_landmarks(frame, hands[0])

        # Display info
        stats = collector.get_stats()
        y_pos = 30

        cv2.putText(frame, f"Current: {collector.current_label or 'None'}",
                    (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        if collecting:
            cv2.putText(frame, f"Collecting: {collect_count}/{samples_per_letter}",
                        (10, y_pos + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # Collect sample
            if hands:
                sample = collector.collect_sample(frame)
                if sample:
                    collect_count += 1

                    if collect_count >= samples_per_letter:
                        collecting = False
                        print(f"Finished collecting {samples_per_letter} samples for '{collector.current_label}'")

            time.sleep(delay)

        cv2.imshow("ASL Data Collection", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        elif key == ord('s'):
            collector.save_samples()
            print("\nCurrent stats:")
            for label, count in stats.items():
                if count > 0:
                    print(f"  {label}: {count} samples")
        elif key == 8:  # Backspace
            collector.set_label('del')
            collecting = True
            collect_count = 0
        elif key == 32:  # Space
            collector.set_label('space')
            collecting = True
            collect_count = 0
        elif key == ord('n'):
            collector.set_label('nothing')
            collecting = True
            collect_count = 0
        elif 65 <= key <= 90 or 97 <= key <= 122:  # A-Z or a-z
            letter = chr(key).upper()
            collector.set_label(letter)
            collecting = True
            collect_count = 0

    cap.release()
    cv2.destroyAllWindows()
    collector.save_samples()
    collector.close()


if __name__ == "__main__":
    run_collection_session()
