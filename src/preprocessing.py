"""
Data Preprocessing Module for ASL Recognition

Handles image loading, preprocessing, and data augmentation
for training the ASL gesture recognition model.
"""

import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Constants
IMG_SIZE = 224  # Standard size for transfer learning models
BATCH_SIZE = 32
NUM_CLASSES = 29  # A-Z + SPACE, DELETE, NOTHING

# Class labels
CLASS_LABELS = [
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
    'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
    'U', 'V', 'W', 'X', 'Y', 'Z', 'del', 'nothing', 'space'
]


def create_data_generators(train_dir, test_dir, img_size=IMG_SIZE, batch_size=BATCH_SIZE):
    """
    Create training and validation data generators with augmentation.
    
    Args:
        train_dir: Path to training data directory
        test_dir: Path to test data directory
        img_size: Target image size (default 224 for transfer learning)
        batch_size: Batch size for training
    
    Returns:
        train_generator, validation_generator, test_generator
    """
    
    # Training data generator with augmentation
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        shear_range=0.1,
        fill_mode='nearest',
        validation_split=0.2  # 20% for validation
    )
    
    # Test/validation data generator (no augmentation)
    test_datagen = ImageDataGenerator(rescale=1./255)
    
    # Training generator
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode='categorical',
        subset='training',
        shuffle=True
    )
    
    # Validation generator
    validation_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation',
        shuffle=False
    )
    
    # Test generator
    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False
    )
    
    return train_generator, validation_generator, test_generator


def preprocess_single_image(image, img_size=IMG_SIZE):
    """
    Preprocess a single image for model inference.
    
    Args:
        image: Input image (numpy array, BGR or RGB)
        img_size: Target size for the model
    
    Returns:
        Preprocessed image ready for model prediction
    """
    import cv2
    
    # Resize image
    if image.shape[:2] != (img_size, img_size):
        image = cv2.resize(image, (img_size, img_size))
    
    # Convert BGR to RGB if needed (OpenCV loads as BGR)
    if len(image.shape) == 3 and image.shape[2] == 3:
        # Normalize to [0, 1]
        image = image.astype(np.float32) / 255.0
    
    # Add batch dimension
    image = np.expand_dims(image, axis=0)
    
    return image


def get_class_label(class_index):
    """
    Get the class label from the predicted class index.
    
    Args:
        class_index: Index of the predicted class
    
    Returns:
        String label for the class
    """
    if 0 <= class_index < len(CLASS_LABELS):
        return CLASS_LABELS[class_index]
    return "Unknown"


def visualize_samples(generator, num_samples=9):
    """
    Visualize sample images from a data generator.
    
    Args:
        generator: Keras data generator
        num_samples: Number of samples to display
    """
    import matplotlib.pyplot as plt
    
    # Get a batch of images
    images, labels = next(generator)
    
    # Create figure
    fig, axes = plt.subplots(3, 3, figsize=(10, 10))
    axes = axes.flatten()
    
    for i in range(min(num_samples, len(images))):
        axes[i].imshow(images[i])
        class_idx = np.argmax(labels[i])
        axes[i].set_title(f"Class: {CLASS_LABELS[class_idx]}")
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Test the preprocessing module
    print("ASL Preprocessing Module")
    print(f"Image Size: {IMG_SIZE}x{IMG_SIZE}")
    print(f"Number of Classes: {NUM_CLASSES}")
    print(f"Class Labels: {CLASS_LABELS}")
