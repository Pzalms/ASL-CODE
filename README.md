# ASL Recognition Web Application

A real-time American Sign Language (ASL) recognition system built with Streamlit and TensorFlow. This application allows users to recognize ASL letters through webcam or image upload, with text-to-speech capabilities and word-building features.

## Features

- **Real-time Webcam Recognition**: Recognize ASL signs in real-time using your webcam
- **Image Upload**: Upload images of ASL signs for recognition
- **Text-to-Speech**: Hear the recognized letters spoken aloud
- **Word Building Mode**: Build words letter by letter with backspace and space support
- **High Accuracy**: Uses EfficientNetB0 model with 89.76% validation accuracy
- **29 Classes**: Recognizes A-Z letters plus 'del', 'nothing', and 'space'

## Installation

### Prerequisites

- Python 3.12 (recommended)
- Webcam (for real-time recognition)
- macOS, Linux, or Windows

### Setup

1. **Clone or navigate to the project directory**:
   ```bash
   cd /path/to/Codework
   ```

2. **Create and activate virtual environment** (if not already done):
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On macOS/Linux
   # or
   .venv\Scripts\activate  # On Windows
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

   This will install:
   - Streamlit (web framework)
   - streamlit-webrtc (webcam support)
   - OpenCV (image processing)
   - TensorFlow (model inference)
   - pyttsx3 & gTTS (text-to-speech)
   - And other required packages

## Running the Application

### Start the Streamlit app:

```bash
streamlit run app.py
```

The application will open in your default web browser at `http://localhost:8501`

### First-time Setup

When you first run the app:
1. Your browser may ask for webcam permissions - click "Allow"
2. The model will load (may take a few seconds on first run)
3. You're ready to start recognizing ASL signs!

## Usage Guide

### Input Methods

#### 📷 Webcam Mode (Recommended)
1. Select "📷 Webcam" in the sidebar
2. Click the "START" button to activate your camera
3. Show ASL signs to the camera
4. Predictions will appear in real-time on the video feed

#### 📁 Upload Image Mode
1. Select "📁 Upload Image" in the sidebar
2. Click "Choose an image..." and select a JPG or PNG file
3. View the prediction results with confidence scores
4. See top 3 predictions with probability bars

### Recognition Modes

#### 🔤 Letter Mode
- Recognizes individual ASL letters
- Displays each prediction independently
- Great for testing and learning

#### 📝 Word Mode
- Build words letter by letter
- Use 'del' sign to backspace (remove last character)
- Use 'space' sign to add spaces between words
- Click "Save Word" to add to your history
- Click "Clear Word" to start over
- Click "Speak Word" to hear the complete word

### Settings

- **Enable Text-to-Speech**: Toggle audio feedback on/off
- **Confidence Threshold**: Adjust minimum confidence (0.5 - 1.0)
  - Higher values = fewer but more confident predictions
  - Lower values = more predictions but potentially less accurate

## ASL Signs Recognized

### Letters
A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P, Q, R, S, T, U, V, W, X, Y, Z

### Special Commands
- **del**: Delete last character (backspace)
- **space**: Add a space
- **nothing**: Neutral position (no prediction)

## Model Information

- **Architecture**: EfficientNetB0 with transfer learning
- **Accuracy**: 89.76% on validation set
- **Input Size**: 224x224 RGB images
- **Framework**: TensorFlow 2.16.2
- **Training Data**: ASL Alphabet dataset (87,000 images)

## Tips for Best Results

1. **Lighting**: Ensure good, even lighting on your hands
2. **Background**: Use a plain, contrasting background
3. **Position**: Center your hand clearly in the frame
4. **Steady Hands**: Hold signs steady for better recognition
5. **Distance**: Keep your hand at a comfortable distance from the camera

## Troubleshooting

### Webcam Not Working
- Check browser permissions for camera access
- Try refreshing the page
- Ensure no other application is using the camera

### Low Accuracy
- Adjust confidence threshold in settings
- Improve lighting conditions
- Make sure signs are clear and well-formed
- Check that background is not cluttered

### Text-to-Speech Not Working (macOS)
If you see warnings about TTS:
```bash
# Install espeak (alternative TTS engine)
brew install espeak
```

### Installation Issues
If you encounter dependency conflicts:
```bash
# Create a fresh virtual environment
python -m venv .venv_new
source .venv_new/bin/activate
pip install -r requirements.txt
```

## Project Structure

```
Codework/
├── app.py                          # Main Streamlit application
├── utils/                          # Utility modules
│   ├── __init__.py
│   ├── config.py                   # Configuration constants
│   ├── model_handler.py            # Model loading and prediction
│   ├── image_processor.py          # Image preprocessing
│   └── tts_handler.py              # Text-to-speech functionality
├── models/
│   └── best_asl_model.keras        # Trained EfficientNetB0 model
├── dataset/                        # Training dataset
├── notebooks/                      # Training notebooks
├── requirements.txt                # Python dependencies
└── README.md                       # This file
```

## Technical Details

### Performance
- **Inference Time**: <100ms per frame
- **Frame Rate**: ~10 FPS in webcam mode
- **Model Size**: 21 MB
- **Memory Usage**: ~500MB during operation

### Preprocessing Pipeline
1. Image resized to 224x224
2. Converted to RGB format
3. EfficientNet preprocessing applied
4. Batch dimension added for inference

## Acknowledgments

- ASL Alphabet dataset
- TensorFlow and Keras teams
- Streamlit community
- EfficientNet architecture (Google Research)

## License

This project is for educational purposes.

## Support

For issues or questions:
1. Check the troubleshooting section above
2. Review the tips for best results
3. Ensure all dependencies are properly installed

---

Built with ❤️ using Streamlit and TensorFlow
