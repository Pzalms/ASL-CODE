"""
Text-to-Speech functionality for ASL Recognition
Uses macOS 'say' command for reliable speech output.
"""

import subprocess
import threading
import queue
import time
import platform


class TTSManager:
    """Thread-safe TTS manager using macOS say command."""

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self._initialized = True
        self._speech_queue = queue.Queue()
        self._last_speech_time = 0
        self._worker_thread = None
        self._running = False
        self._is_macos = platform.system() == "Darwin"
        self._current_process = None

        self._start_worker()

    def _start_worker(self):
        """Start background worker thread."""
        self._running = True
        self._worker_thread = threading.Thread(target=self._worker, daemon=True)
        self._worker_thread.start()

    def _worker(self):
        """Background worker that processes speech queue."""
        while self._running:
            try:
                text = self._speech_queue.get(timeout=0.5)
                if text:
                    self._speak_sync(text)
                self._speech_queue.task_done()
            except queue.Empty:
                continue
            except Exception:
                pass

    def _speak_sync(self, text):
        """Speak text synchronously using system command."""
        try:
            if self._is_macos:
                # Use macOS 'say' command - very reliable
                self._current_process = subprocess.Popen(
                    ["say", "-r", "180", text],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )
                self._current_process.wait()
                self._current_process = None
            else:
                # Fallback to pyttsx3 for other platforms
                try:
                    import pyttsx3
                    engine = pyttsx3.init()
                    engine.setProperty('rate', 160)
                    engine.say(text)
                    engine.runAndWait()
                    engine.stop()
                except Exception:
                    pass
        except Exception as e:
            print(f"TTS error: {e}")

    def stop_current(self):
        """Stop current speech."""
        if self._current_process:
            try:
                self._current_process.terminate()
            except Exception:
                pass

    def speak(self, text, cooldown=0.3):
        """Queue text for speech with cooldown."""
        if not text:
            return False

        current_time = time.time()
        if current_time - self._last_speech_time < cooldown:
            return False

        # Handle special characters
        speech_text = text
        if text == 'del':
            speech_text = "delete"
        elif text == 'space':
            speech_text = "space"
        elif text == 'nothing':
            return False

        # Clear queue to prevent buildup
        self._clear_queue()

        self._speech_queue.put(speech_text)
        self._last_speech_time = current_time
        return True

    def speak_word(self, word):
        """Speak a full word/sentence."""
        if not word or not word.strip():
            return False

        self._clear_queue()
        self._speech_queue.put(word.strip())
        return True

    def _clear_queue(self):
        """Clear the speech queue."""
        while not self._speech_queue.empty():
            try:
                self._speech_queue.get_nowait()
            except queue.Empty:
                break

    @property
    def is_available(self):
        return self._is_macos or True  # Always try


# Global TTS manager instance
_tts_manager = None


def get_tts_manager():
    """Get or create TTS manager singleton."""
    global _tts_manager
    if _tts_manager is None:
        _tts_manager = TTSManager()
    return _tts_manager


def speak_text(text, cooldown=0.3):
    """Speak text using TTS with cooldown mechanism."""
    try:
        import streamlit as st
        if not st.session_state.get('tts_enabled', True):
            return False
    except Exception:
        pass

    manager = get_tts_manager()
    return manager.speak(text, cooldown)


def speak_word(word):
    """Speak an entire word or sentence."""
    try:
        import streamlit as st
        if not st.session_state.get('tts_enabled', True):
            return False
    except Exception:
        pass

    manager = get_tts_manager()
    return manager.speak_word(word)


def speak_letter(letter):
    """Speak a single letter with short cooldown."""
    return speak_text(letter, cooldown=0.2)
