"""
Prediction Module for ASL Recognition

Handles model loading and real-time prediction from camera frames.
"""

import os
import numpy as np
import cv2
from tensorflow.keras.models import load_model

# Import preprocessing utilities
from .preprocessing import preprocess_single_image, CLASS_LABELS, IMG_SIZE


class ASLPredictor:
    """
    ASL Gesture Predictor class for real-time inference.
    """
    
    def __init__(self, model_path=None):
        """
        Initialize the predictor with a trained model.
        
        Args:
            model_path: Path to the trained model file (.h5)
        """
        self.model = None
        self.model_path = model_path
        self.class_labels = CLASS_LABELS
        
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
    
    def load_model(self, model_path):
        """
        Load a trained model from file.
        
        Args:
            model_path: Path to the .h5 model file
        """
        try:
            self.model = load_model(model_path)
            self.model_path = model_path
            print(f"Model loaded successfully from: {model_path}")
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
    
    def predict(self, image):
        """
        Make a prediction on a single image.
        
        Args:
            image: Input image (numpy array, BGR from OpenCV)
        
        Returns:
            Dictionary with 'label', 'confidence', and 'all_probs'
        """
        if self.model is None:
            raise ValueError("No model loaded. Call load_model() first.")
        
        # Preprocess the image
        processed = preprocess_single_image(image)
        
        # Get predictions
        predictions = self.model.predict(processed, verbose=0)
        
        # Get top prediction
        class_idx = np.argmax(predictions[0])
        confidence = float(predictions[0][class_idx])
        label = self.class_labels[class_idx]
        
        return {
            'label': label,
            'confidence': confidence,
            'class_index': int(class_idx),
            'all_probs': predictions[0].tolist()
        }
    
    def predict_top_k(self, image, k=3):
        """
        Get top-k predictions for an image.
        
        Args:
            image: Input image
            k: Number of top predictions to return
        
        Returns:
            List of (label, confidence) tuples
        """
        if self.model is None:
            raise ValueError("No model loaded. Call load_model() first.")
        
        processed = preprocess_single_image(image)
        predictions = self.model.predict(processed, verbose=0)[0]
        
        # Get top-k indices
        top_indices = np.argsort(predictions)[-k:][::-1]
        
        results = []
        for idx in top_indices:
            results.append({
                'label': self.class_labels[idx],
                'confidence': float(predictions[idx])
            })
        
        return results


def create_predictor(model_path="models/best_asl_model.h5"):
    """
    Factory function to create an ASL predictor.
    
    Args:
        model_path: Path to the trained model
    
    Returns:
        ASLPredictor instance
    """
    return ASLPredictor(model_path)


if __name__ == "__main__":
    # Test the prediction module
    print("ASL Prediction Module")
    print(f"Class Labels: {CLASS_LABELS}")
    print(f"Number of Classes: {len(CLASS_LABELS)}")
