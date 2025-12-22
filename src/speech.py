"""
Text-to-Speech Module for ASL Recognition

Converts recognized sign language gestures to audible speech.
Uses gTTS (Google Text-to-Speech) as primary and pyttsx3 as fallback.
"""

import os
import tempfile
import threading
from io import BytesIO

# Try to import gTTS (requires internet)
try:
    from gtts import gTTS
    GTTS_AVAILABLE = True
except ImportError:
    GTTS_AVAILABLE = False

# Try to import pyttsx3 (offline)
try:
    import pyttsx3
    PYTTSX3_AVAILABLE = True
except ImportError:
    PYTTSX3_AVAILABLE = False


class SpeechEngine:
    """
    Text-to-Speech engine for ASL recognition output.
    """
    
    def __init__(self, use_gtts=True, language='en'):
        """
        Initialize the speech engine.
        
        Args:
            use_gtts: Prefer gTTS (online) over pyttsx3 (offline)
            language: Language code for speech synthesis
        """
        self.language = language
        self.use_gtts = use_gtts and GTTS_AVAILABLE
        self.temp_dir = tempfile.gettempdir()
        
        # Initialize pyttsx3 engine if available
        self.pyttsx_engine = None
        if PYTTSX3_AVAILABLE:
            try:
                self.pyttsx_engine = pyttsx3.init()
                # Set properties
                self.pyttsx_engine.setProperty('rate', 150)
                self.pyttsx_engine.setProperty('volume', 0.9)
            except Exception as e:
                print(f"Could not initialize pyttsx3: {e}")
    
    def text_to_speech_gtts(self, text):
        """
        Convert text to speech using Google TTS.
        
        Args:
            text: Text to convert to speech
        
        Returns:
            BytesIO object containing MP3 audio data
        """
        if not GTTS_AVAILABLE:
            raise RuntimeError("gTTS is not installed")
        
        tts = gTTS(text=text, lang=self.language, slow=False)
        audio_buffer = BytesIO()
        tts.write_to_fp(audio_buffer)
        audio_buffer.seek(0)
        return audio_buffer
    
    def text_to_speech_pyttsx(self, text):
        """
        Convert text to speech using pyttsx3 (offline).
        
        Args:
            text: Text to speak
        """
        if not PYTTSX3_AVAILABLE or self.pyttsx_engine is None:
            raise RuntimeError("pyttsx3 is not available")
        
        self.pyttsx_engine.say(text)
        self.pyttsx_engine.runAndWait()
    
    def save_to_file(self, text, filepath):
        """
        Save speech to an audio file.
        
        Args:
            text: Text to convert
            filepath: Output file path
        """
        if self.use_gtts and GTTS_AVAILABLE:
            tts = gTTS(text=text, lang=self.language, slow=False)
            tts.save(filepath)
        elif PYTTSX3_AVAILABLE and self.pyttsx_engine:
            self.pyttsx_engine.save_to_file(text, filepath)
            self.pyttsx_engine.runAndWait()
        else:
            raise RuntimeError("No TTS engine available")
    
    def speak(self, text):
        """
        Speak the given text.
        
        Args:
            text: Text to speak aloud
        """
        if not text or text.strip() == '':
            return
        
        try:
            if self.use_gtts and GTTS_AVAILABLE:
                # For gTTS, save to temp file and play
                temp_file = os.path.join(self.temp_dir, 'asl_speech.mp3')
                self.save_to_file(text, temp_file)
                # Note: Playing audio requires additional setup
                # For Flask, we'll serve the audio file instead
                return temp_file
            elif PYTTSX3_AVAILABLE and self.pyttsx_engine:
                self.text_to_speech_pyttsx(text)
            else:
                print(f"TTS: {text}")  # Fallback to print
        except Exception as e:
            print(f"Speech error: {e}")
    
    def get_audio_bytes(self, text):
        """
        Get audio as bytes for streaming.
        
        Args:
            text: Text to convert
        
        Returns:
            Audio bytes (MP3 format if using gTTS)
        """
        if self.use_gtts and GTTS_AVAILABLE:
            buffer = self.text_to_speech_gtts(text)
            return buffer.read()
        else:
            # For pyttsx3, save to temp and read
            temp_file = os.path.join(self.temp_dir, 'asl_speech_temp.wav')
            self.save_to_file(text, temp_file)
            with open(temp_file, 'rb') as f:
                return f.read()


# Global speech engine instance
_speech_engine = None


def get_speech_engine():
    """
    Get or create the global speech engine.
    """
    global _speech_engine
    if _speech_engine is None:
        _speech_engine = SpeechEngine()
    return _speech_engine


def speak_text(text):
    """
    Convenience function to speak text.
    
    Args:
        text: Text to speak
    """
    engine = get_speech_engine()
    return engine.speak(text)


def get_audio_response(text):
    """
    Get audio bytes for a text string.
    
    Args:
        text: Text to convert to speech
    
    Returns:
        Audio bytes
    """
    engine = get_speech_engine()
    return engine.get_audio_bytes(text)


if __name__ == "__main__":
    print("ASL Speech Module")
    print(f"gTTS available: {GTTS_AVAILABLE}")
    print(f"pyttsx3 available: {PYTTSX3_AVAILABLE}")
    
    # Test speech
    engine = SpeechEngine()
    print("\nTesting speech synthesis...")
    engine.speak("Hello, this is a test of the ASL recognition speech system.")
