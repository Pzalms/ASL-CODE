"""
Hand Detection and Landmark Extraction using MediaPipe Tasks API

This module provides real-time hand detection and landmark extraction
for ASL finger spelling recognition.
"""

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from dataclasses import dataclass
from typing import Optional, List, Tuple
from pathlib import Path


@dataclass
class HandLandmarks:
    """Container for hand landmark data."""
    landmarks: np.ndarray  # Shape: (21, 3) - 21 landmarks, x/y/z each
    handedness: str  # 'Left' or 'Right'
    confidence: float
    bbox: Tuple[int, int, int, int]  # x, y, w, h


class HandDetector:
    """
    MediaPipe Tasks-based hand detector for ASL recognition.

    Extracts 21 hand landmarks per hand, which can be used
    for gesture classification.
    """

    # MediaPipe hand landmark indices
    WRIST = 0
    THUMB_CMC, THUMB_MCP, THUMB_IP, THUMB_TIP = 1, 2, 3, 4
    INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP = 5, 6, 7, 8
    MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP = 9, 10, 11, 12
    RING_MCP, RING_PIP, RING_DIP, RING_TIP = 13, 14, 15, 16
    PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP = 17, 18, 19, 20

    # Default model path
    MODEL_PATH = Path(__file__).parent.parent / "models" / "hand_landmarker.task"

    def __init__(
        self,
        max_hands: int = 1,
        min_detection_confidence: float = 0.7,
        min_tracking_confidence: float = 0.5,
        static_image_mode: bool = False,
        model_path: Optional[str] = None
    ):
        """
        Initialize the hand detector.

        Args:
            max_hands: Maximum number of hands to detect
            min_detection_confidence: Minimum confidence for detection
            min_tracking_confidence: Minimum confidence for tracking
            static_image_mode: If True, treats each image independently
            model_path: Path to hand_landmarker.task model file
        """
        self.max_hands = max_hands
        self.min_detection_confidence = min_detection_confidence
        self.static_image_mode = static_image_mode

        # Use provided model path or default
        model_file = model_path or str(self.MODEL_PATH)

        # Configure the hand landmarker
        base_options = python.BaseOptions(model_asset_path=model_file)

        if static_image_mode:
            running_mode = vision.RunningMode.IMAGE
        else:
            running_mode = vision.RunningMode.IMAGE  # Use IMAGE mode for simplicity

        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=running_mode,
            num_hands=max_hands,
            min_hand_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )

        self.detector = vision.HandLandmarker.create_from_options(options)

    def detect(self, image: np.ndarray) -> List[HandLandmarks]:
        """
        Detect hands and extract landmarks from an image.

        Args:
            image: BGR image from OpenCV

        Returns:
            List of HandLandmarks objects for each detected hand
        """
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w, _ = image.shape

        # Create MediaPipe Image
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)

        # Detect hands
        results = self.detector.detect(mp_image)

        hands_data = []

        if results.hand_landmarks:
            for idx, hand_landmarks in enumerate(results.hand_landmarks):
                # Extract landmarks as numpy array
                landmarks = np.array([
                    [lm.x, lm.y, lm.z] for lm in hand_landmarks
                ])

                # Get handedness
                if results.handedness and idx < len(results.handedness):
                    handedness_info = results.handedness[idx][0]
                    hand_label = handedness_info.category_name
                    confidence = handedness_info.score
                else:
                    hand_label = "Unknown"
                    confidence = 0.0

                # Calculate bounding box
                x_coords = landmarks[:, 0] * w
                y_coords = landmarks[:, 1] * h
                bbox = (
                    int(x_coords.min()),
                    int(y_coords.min()),
                    int(x_coords.max() - x_coords.min()),
                    int(y_coords.max() - y_coords.min())
                )

                hands_data.append(HandLandmarks(
                    landmarks=landmarks,
                    handedness=hand_label,
                    confidence=confidence,
                    bbox=bbox
                ))

        return hands_data

    def extract_features(self, landmarks: np.ndarray) -> np.ndarray:
        """
        Extract normalized features from hand landmarks.

        Normalizes landmarks relative to wrist position and hand size
        for scale/position invariance.

        Args:
            landmarks: Raw landmarks array (21, 3)

        Returns:
            Normalized feature vector (63,) - flattened landmarks
        """
        # Center on wrist
        centered = landmarks - landmarks[self.WRIST]

        # Normalize by hand size (distance from wrist to middle finger MCP)
        hand_size = np.linalg.norm(centered[self.MIDDLE_MCP])
        if hand_size > 0:
            normalized = centered / hand_size
        else:
            normalized = centered

        # Flatten to 1D feature vector
        return normalized.flatten()

    def extract_angles(self, landmarks: np.ndarray) -> np.ndarray:
        """
        Extract finger joint angles from landmarks.

        Returns angles for each finger joint, which are more
        robust to hand orientation.

        Args:
            landmarks: Raw landmarks array (21, 3)

        Returns:
            Angle features array (15,) - 3 angles per finger
        """
        angles = []

        # Finger definitions: (MCP, PIP, DIP, TIP)
        fingers = [
            (self.INDEX_MCP, self.INDEX_PIP, self.INDEX_DIP, self.INDEX_TIP),
            (self.MIDDLE_MCP, self.MIDDLE_PIP, self.MIDDLE_DIP, self.MIDDLE_TIP),
            (self.RING_MCP, self.RING_PIP, self.RING_DIP, self.RING_TIP),
            (self.PINKY_MCP, self.PINKY_PIP, self.PINKY_DIP, self.PINKY_TIP),
        ]

        # Thumb is special
        thumb = (self.THUMB_CMC, self.THUMB_MCP, self.THUMB_IP, self.THUMB_TIP)

        for finger in [thumb] + fingers:
            for i in range(len(finger) - 2):
                p1 = landmarks[finger[i]]
                p2 = landmarks[finger[i + 1]]
                p3 = landmarks[finger[i + 2]]

                v1 = p1 - p2
                v2 = p3 - p2

                # Calculate angle
                cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
                angle = np.arccos(np.clip(cos_angle, -1, 1))
                angles.append(angle)

        return np.array(angles)

    def get_combined_features(self, landmarks: np.ndarray) -> np.ndarray:
        """
        Get combined feature vector (positions + angles).

        Args:
            landmarks: Raw landmarks array (21, 3)

        Returns:
            Combined feature vector (78,) - 63 positions + 15 angles
        """
        positions = self.extract_features(landmarks)
        angles = self.extract_angles(landmarks)
        return np.concatenate([positions, angles])

    def draw_landmarks(
        self,
        image: np.ndarray,
        hand_data: HandLandmarks,
        draw_bbox: bool = True
    ) -> np.ndarray:
        """
        Draw hand landmarks and connections on image.

        Args:
            image: BGR image
            hand_data: HandLandmarks object
            draw_bbox: Whether to draw bounding box

        Returns:
            Image with drawings
        """
        img_copy = image.copy()
        h, w, _ = img_copy.shape

        # Hand connections for drawing
        HAND_CONNECTIONS = [
            (0, 1), (1, 2), (2, 3), (3, 4),  # Thumb
            (0, 5), (5, 6), (6, 7), (7, 8),  # Index
            (0, 9), (9, 10), (10, 11), (11, 12),  # Middle
            (0, 13), (13, 14), (14, 15), (15, 16),  # Ring
            (0, 17), (17, 18), (18, 19), (19, 20),  # Pinky
            (5, 9), (9, 13), (13, 17)  # Palm
        ]

        # Draw connections
        for start_idx, end_idx in HAND_CONNECTIONS:
            start_point = (
                int(hand_data.landmarks[start_idx][0] * w),
                int(hand_data.landmarks[start_idx][1] * h)
            )
            end_point = (
                int(hand_data.landmarks[end_idx][0] * w),
                int(hand_data.landmarks[end_idx][1] * h)
            )
            cv2.line(img_copy, start_point, end_point, (0, 255, 0), 2)

        # Draw landmarks
        for idx, lm in enumerate(hand_data.landmarks):
            cx, cy = int(lm[0] * w), int(lm[1] * h)
            cv2.circle(img_copy, (cx, cy), 5, (255, 0, 0), -1)

        # Draw bounding box
        if draw_bbox:
            x, y, bw, bh = hand_data.bbox
            cv2.rectangle(img_copy, (x, y), (x + bw, y + bh), (0, 255, 0), 2)
            cv2.putText(
                img_copy,
                f"{hand_data.handedness}: {hand_data.confidence:.0%}",
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

        return img_copy

    def close(self):
        """Release resources."""
        if hasattr(self, 'detector'):
            self.detector.close()


# Convenience function for quick detection
def detect_hand_landmarks(image: np.ndarray) -> Optional[np.ndarray]:
    """
    Quick function to detect hand and return normalized features.

    Args:
        image: BGR image

    Returns:
        Feature vector (78,) or None if no hand detected
    """
    detector = HandDetector(max_hands=1, static_image_mode=True)
    hands = detector.detect(image)
    detector.close()

    if hands:
        return detector.get_combined_features(hands[0].landmarks)
    return None
