"""
Transfer Learning Model Module for ASL Recognition

Provides factory functions to create various pre-trained models
with custom classification heads for ASL gesture recognition.
"""

from tensorflow.keras.applications import (
    EfficientNetB0,
    MobileNetV2,
    VGG16,
    Xception
)
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Dense,
    Dropout,
    GlobalAveragePooling2D,
    Input
)
from tensorflow.keras.optimizers import Adam

# Constants
IMG_SIZE = 224
NUM_CLASSES = 29


def create_efficientnet_model(num_classes=NUM_CLASSES, img_size=IMG_SIZE, trainable_base=False):
    """
    Create EfficientNetB0 model with custom classification head.
    
    EfficientNet offers the best accuracy/efficiency trade-off.
    Parameters: ~5.3M
    """
    base_model = EfficientNetB0(
        weights='imagenet',
        include_top=False,
        input_shape=(img_size, img_size, 3)
    )
    base_model.trainable = trainable_base
    
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    outputs = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs=base_model.input, outputs=outputs, name='EfficientNetB0_ASL')
    return model


def create_mobilenet_model(num_classes=NUM_CLASSES, img_size=IMG_SIZE, trainable_base=False):
    """
    Create MobileNetV2 model with custom classification head.
    
    MobileNet is lightweight and fast for inference.
    Parameters: ~3.4M
    """
    base_model = MobileNetV2(
        weights='imagenet',
        include_top=False,
        input_shape=(img_size, img_size, 3)
    )
    base_model.trainable = trainable_base
    
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    outputs = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs=base_model.input, outputs=outputs, name='MobileNetV2_ASL')
    return model


def create_vgg16_model(num_classes=NUM_CLASSES, img_size=IMG_SIZE, trainable_base=False):
    """
    Create VGG16 model with custom classification head.
    
    VGG16 is a deep, well-studied architecture.
    Parameters: ~138M (large)
    """
    base_model = VGG16(
        weights='imagenet',
        include_top=False,
        input_shape=(img_size, img_size, 3)
    )
    base_model.trainable = trainable_base
    
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    outputs = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs=base_model.input, outputs=outputs, name='VGG16_ASL')
    return model


def create_xception_model(num_classes=NUM_CLASSES, img_size=IMG_SIZE, trainable_base=False):
    """
    Create Xception model with custom classification head.
    
    Xception provides excellent accuracy with depthwise separable convolutions.
    Parameters: ~22.9M
    """
    base_model = Xception(
        weights='imagenet',
        include_top=False,
        input_shape=(img_size, img_size, 3)
    )
    base_model.trainable = trainable_base
    
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    outputs = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs=base_model.input, outputs=outputs, name='Xception_ASL')
    return model


def get_model_by_name(model_name, num_classes=NUM_CLASSES, trainable_base=False):
    """
    Factory function to get model by name.
    
    Args:
        model_name: One of 'efficientnet', 'mobilenet', 'vgg16', 'xception'
        num_classes: Number of output classes
        trainable_base: Whether to train the base model layers
    
    Returns:
        Compiled Keras model
    """
    model_map = {
        'efficientnet': create_efficientnet_model,
        'mobilenet': create_mobilenet_model,
        'vgg16': create_vgg16_model,
        'xception': create_xception_model
    }
    
    model_name = model_name.lower()
    if model_name not in model_map:
        raise ValueError(f"Unknown model: {model_name}. Choose from: {list(model_map.keys())}")
    
    model = model_map[model_name](num_classes=num_classes, trainable_base=trainable_base)
    return model


def compile_model(model, learning_rate=0.001):
    """
    Compile model with Adam optimizer and categorical crossentropy.
    
    Args:
        model: Keras model to compile
        learning_rate: Learning rate for Adam optimizer
    
    Returns:
        Compiled model
    """
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


def get_all_models(num_classes=NUM_CLASSES, compile_models=True):
    """
    Get all available models for comparison.
    
    Returns:
        Dictionary of model_name -> compiled model
    """
    models = {}
    for name in ['efficientnet', 'mobilenet', 'vgg16', 'xception']:
        model = get_model_by_name(name, num_classes=num_classes)
        if compile_models:
            model = compile_model(model)
        models[name] = model
    return models


if __name__ == "__main__":
    # Test model creation
    print("Testing ASL Model Creation...")
    
    for name in ['efficientnet', 'mobilenet', 'vgg16', 'xception']:
        model = get_model_by_name(name)
        model = compile_model(model)
        print(f"\n{model.name}:")
        print(f"  Total params: {model.count_params():,}")
        print(f"  Trainable params: {sum(w.numpy().size for w in model.trainable_weights):,}")
