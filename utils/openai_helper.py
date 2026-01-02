"""
OpenAI integration for intelligent word correction in ASL recognition.
Uses GPT to correct and predict words from partial finger-spelled input.
"""

import os
from typing import Optional, List, Tuple


class WordCorrector:
    """OpenAI-powered word correction for finger spelling."""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key
        self.client = None
        self._last_correction = None
        self._cache = {}

        if api_key:
            self._init_client()

    def _init_client(self):
        """Initialize OpenAI client."""
        try:
            from openai import OpenAI
            self.client = OpenAI(api_key=self.api_key)
        except ImportError:
            print("OpenAI package not installed. Run: pip install openai")
            self.client = None
        except Exception as e:
            print(f"Failed to initialize OpenAI: {e}")
            self.client = None

    def set_api_key(self, api_key: str):
        """Update API key and reinitialize client."""
        self.api_key = api_key
        self._cache = {}
        self._init_client()

    @property
    def is_available(self) -> bool:
        """Check if OpenAI is available."""
        return self.client is not None and self.api_key is not None

    def correct_word(self, partial_word: str, context: str = "") -> Tuple[str, List[str]]:
        """
        Correct/complete a partially spelled word.

        Args:
            partial_word: The letters spelled so far (may have errors)
            context: Optional context from previous words

        Returns:
            Tuple of (best_correction, list_of_suggestions)
        """
        if not self.is_available or not partial_word or len(partial_word) < 2:
            return partial_word, []

        # Check cache
        cache_key = f"{partial_word}:{context}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        try:
            prompt = self._build_prompt(partial_word, context)

            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": """You are a word correction assistant for ASL finger spelling recognition.
Your task is to correct misspelled words from finger spelling input where letters may be misrecognized.

Common ASL recognition errors:
- Similar hand shapes confused: M/N, U/V, A/S/E, I/J, K/V
- Quick transitions missed
- Letters dropped or doubled

Respond with ONLY a JSON object in this exact format:
{"best": "corrected_word", "alternatives": ["alt1", "alt2"]}

Rules:
- Return common English words only
- If the input looks like an incomplete word, predict the most likely completion
- Maximum 3 alternatives
- Keep corrections close to the original letters when possible"""
                    },
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=100
            )

            result = response.choices[0].message.content.strip()

            # Parse JSON response
            import json
            data = json.loads(result)
            best = data.get("best", partial_word)
            alternatives = data.get("alternatives", [])

            # Cache result
            self._cache[cache_key] = (best, alternatives)
            self._last_correction = best

            return best, alternatives

        except Exception as e:
            print(f"OpenAI correction error: {e}")
            return partial_word, []

    def _build_prompt(self, partial_word: str, context: str) -> str:
        """Build the correction prompt."""
        prompt = f"Correct this finger-spelled word: '{partial_word}'"
        if context:
            prompt += f"\nPrevious words for context: {context}"
        return prompt

    def predict_next_word(self, sentence: str) -> List[str]:
        """Predict likely next words based on context."""
        if not self.is_available or not sentence:
            return []

        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": "Predict the 3 most likely next words for this sentence. Respond with ONLY a JSON array of words: [\"word1\", \"word2\", \"word3\"]"
                    },
                    {"role": "user", "content": f"Sentence: {sentence}"}
                ],
                temperature=0.5,
                max_tokens=50
            )

            import json
            result = response.choices[0].message.content.strip()
            return json.loads(result)

        except Exception:
            return []

    def get_instant_correction(self, letters: str) -> Optional[str]:
        """
        Fast correction for real-time use.
        Uses a simpler prompt for lower latency.
        """
        if not self.is_available or len(letters) < 3:
            return None

        # Check cache first
        if letters in self._cache:
            return self._cache[letters][0]

        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": "You correct ASL finger-spelled words. Reply with ONLY the corrected word, nothing else."
                    },
                    {"role": "user", "content": f"Correct: {letters}"}
                ],
                temperature=0.2,
                max_tokens=20
            )

            corrected = response.choices[0].message.content.strip().lower()

            # Basic validation
            if corrected and corrected.isalpha() and len(corrected) <= len(letters) + 3:
                self._cache[letters] = (corrected, [])
                return corrected

            return None

        except Exception:
            return None


# Global instance
_word_corrector = None


def get_word_corrector(api_key: Optional[str] = None) -> WordCorrector:
    """Get or create the word corrector instance."""
    global _word_corrector
    if _word_corrector is None:
        _word_corrector = WordCorrector(api_key)
    elif api_key and api_key != _word_corrector.api_key:
        _word_corrector.set_api_key(api_key)
    return _word_corrector
