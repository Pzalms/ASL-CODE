"""
Finger Spelling Recognition System

Uses hand landmarks + LSTM for real-time finger spelling to text conversion.
Includes stability detection and word formation.
"""

import numpy as np
from collections import deque
from typing import Optional, List, Tuple
import threading


class LetterBuffer:
    """
    Manages letter detection with stability checking.
    Only registers a letter when it's held steady for multiple frames.
    """

    def __init__(self, stability_frames: int = 5, cooldown_frames: int = 10):
        """
        Args:
            stability_frames: Frames needed to confirm a letter
            cooldown_frames: Frames to wait before detecting same letter again
        """
        self.stability_frames = stability_frames
        self.cooldown_frames = cooldown_frames

        self.history = deque(maxlen=stability_frames)
        self.last_confirmed = None
        self.cooldown_counter = 0
        self.current_word = ""
        self.words = []

    def update(self, letter: str, confidence: float) -> Optional[str]:
        """
        Update with new prediction and return confirmed letter if stable.

        Args:
            letter: Predicted letter
            confidence: Prediction confidence

        Returns:
            Confirmed letter if stable, None otherwise
        """
        # Decrease cooldown
        if self.cooldown_counter > 0:
            self.cooldown_counter -= 1

        # Add to history
        self.history.append((letter, confidence))

        # Check if we have enough history
        if len(self.history) < self.stability_frames:
            return None

        # Check if all recent predictions are the same
        letters = [h[0] for h in self.history]
        if len(set(letters)) == 1:
            confirmed_letter = letters[0]

            # Check cooldown (prevent repeated same letter)
            if confirmed_letter == self.last_confirmed and self.cooldown_counter > 0:
                return None

            # Confirm the letter
            self.last_confirmed = confirmed_letter
            self.cooldown_counter = self.cooldown_frames
            self.history.clear()

            return confirmed_letter

        return None

    def add_to_word(self, letter: str):
        """Add confirmed letter to current word."""
        if letter == 'space':
            if self.current_word:
                self.words.append(self.current_word)
                self.current_word = ""
        elif letter == 'del':
            self.current_word = self.current_word[:-1]
        elif letter == 'nothing':
            pass  # Ignore
        else:
            self.current_word += letter

    def get_current_text(self) -> str:
        """Get full text including completed words and current word."""
        all_words = self.words + ([self.current_word] if self.current_word else [])
        return " ".join(all_words)

    def clear(self):
        """Clear all text."""
        self.current_word = ""
        self.words = []
        self.history.clear()
        self.last_confirmed = None


class FingerSpellingRecognizer:
    """
    Complete finger spelling recognition system.

    Combines:
    - Hand landmark detection
    - Letter classification
    - Stability detection
    - Word formation
    """

    # ASL Alphabet labels
    LABELS = [
        'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
        'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
        'del', 'nothing', 'space'
    ]

    def __init__(
        self,
        model=None,
        stability_frames: int = 5,
        confidence_threshold: float = 0.7
    ):
        """
        Args:
            model: Trained classification model
            stability_frames: Frames needed to confirm letter
            confidence_threshold: Minimum confidence for valid prediction
        """
        self.model = model
        self.confidence_threshold = confidence_threshold
        self.letter_buffer = LetterBuffer(stability_frames=stability_frames)

        # Current state
        self.current_prediction = None
        self.current_confidence = 0.0
        self.is_stable = False

        # Lock for thread safety
        self.lock = threading.Lock()

    def predict(self, features: np.ndarray) -> Tuple[str, float]:
        """
        Predict letter from hand features.

        Args:
            features: Normalized hand landmark features

        Returns:
            (predicted_letter, confidence)
        """
        if self.model is None:
            return 'nothing', 0.0

        # Ensure correct shape
        if len(features.shape) == 1:
            features = features.reshape(1, -1)

        # Get prediction
        predictions = self.model.predict(features, verbose=0)
        pred_idx = np.argmax(predictions[0])
        confidence = float(predictions[0][pred_idx])

        letter = self.LABELS[pred_idx] if pred_idx < len(self.LABELS) else 'nothing'

        return letter, confidence

    def process_frame(self, features: np.ndarray) -> dict:
        """
        Process a single frame and update state.

        Args:
            features: Hand landmark features

        Returns:
            Dict with current state info
        """
        with self.lock:
            # Get prediction
            letter, confidence = self.predict(features)

            self.current_prediction = letter
            self.current_confidence = confidence

            confirmed = None
            self.is_stable = False

            # Only process if confidence is high enough
            if confidence >= self.confidence_threshold:
                confirmed = self.letter_buffer.update(letter, confidence)

                if confirmed:
                    self.is_stable = True
                    self.letter_buffer.add_to_word(confirmed)

            return {
                'prediction': letter,
                'confidence': confidence,
                'confirmed': confirmed,
                'is_stable': self.is_stable,
                'current_word': self.letter_buffer.current_word,
                'full_text': self.letter_buffer.get_current_text()
            }

    def get_text(self) -> str:
        """Get current recognized text."""
        with self.lock:
            return self.letter_buffer.get_current_text()

    def clear_text(self):
        """Clear recognized text."""
        with self.lock:
            self.letter_buffer.clear()

    def set_model(self, model):
        """Set or update the classification model."""
        with self.lock:
            self.model = model


class SpellingCorrector:
    """
    Simple spelling correction for recognized text.
    Uses edit distance to suggest corrections.
    """

    def __init__(self, dictionary_file: Optional[str] = None):
        """
        Args:
            dictionary_file: Path to word list file
        """
        # Common English words
        self.dictionary = {
            'hello', 'world', 'thank', 'you', 'please', 'help', 'yes', 'no',
            'good', 'bad', 'love', 'hate', 'want', 'need', 'like', 'name',
            'what', 'where', 'when', 'why', 'how', 'who', 'can', 'will',
            'food', 'water', 'home', 'work', 'school', 'friend', 'family',
            'mother', 'father', 'sister', 'brother', 'baby', 'child',
            'happy', 'sad', 'angry', 'tired', 'hungry', 'thirsty',
            'morning', 'afternoon', 'evening', 'night', 'today', 'tomorrow',
            'monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday'
        }

        if dictionary_file:
            self._load_dictionary(dictionary_file)

    def _load_dictionary(self, file_path: str):
        """Load words from file."""
        try:
            with open(file_path, 'r') as f:
                words = {line.strip().lower() for line in f if line.strip()}
                self.dictionary.update(words)
        except Exception:
            pass

    def _edit_distance(self, s1: str, s2: str) -> int:
        """Calculate Levenshtein distance between two strings."""
        m, n = len(s1), len(s2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]

        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if s1[i-1] == s2[j-1]:
                    dp[i][j] = dp[i-1][j-1]
                else:
                    dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])

        return dp[m][n]

    def suggest(self, word: str, max_suggestions: int = 3) -> List[str]:
        """
        Get spelling suggestions for a word.

        Args:
            word: Input word
            max_suggestions: Maximum number of suggestions

        Returns:
            List of suggested corrections
        """
        word_lower = word.lower()

        # If word is in dictionary, return it
        if word_lower in self.dictionary:
            return [word]

        # Find closest matches
        candidates = []
        for dict_word in self.dictionary:
            distance = self._edit_distance(word_lower, dict_word)
            if distance <= 2:  # Allow up to 2 edits
                candidates.append((distance, dict_word))

        # Sort by distance and return top suggestions
        candidates.sort(key=lambda x: x[0])
        return [c[1] for c in candidates[:max_suggestions]]

    def correct_text(self, text: str) -> str:
        """
        Attempt to correct an entire text string.

        Args:
            text: Input text

        Returns:
            Corrected text
        """
        words = text.split()
        corrected = []

        for word in words:
            suggestions = self.suggest(word)
            if suggestions:
                corrected.append(suggestions[0])
            else:
                corrected.append(word)

        return " ".join(corrected)
