"""
Model Architecture Module for Plant Health Detection System
Implements ResNet50-based transfer learning model
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.optimizers import Adam
import config
import numpy as np


def build_model(num_classes, freeze_base=True):
    """
    Build ResNet50-based transfer learning model
    
    Args:
        num_classes: Number of output classes
        freeze_base: Whether to freeze ResNet50 base layers
        
    Returns:
        keras.Model: Compiled model ready for training
    """
    print(f"\n{'='*60}")
    print("🏗️  Building Model Architecture")
    print(f"{'='*60}")
    
    # Load pretrained ResNet50
    base_model = ResNet50(
        weights=config.WEIGHTS,
        include_top=False,
        input_shape=(config.IMG_HEIGHT, config.IMG_WIDTH, config.IMG_CHANNELS)
    )
    
    # Freeze base model layers if specified
    if freeze_base:
        base_model.trainable = False
        print(f"✅ Base model (ResNet50) loaded with frozen weights")
    else:
        base_model.trainable = True
        print(f"✅ Base model (ResNet50) loaded with trainable weights")
    
    # Build custom classification head
    inputs = keras.Input(shape=(config.IMG_HEIGHT, config.IMG_WIDTH, config.IMG_CHANNELS))
    
    # ResNet50 base
    x = base_model(inputs, training=False)
    
    # Global average pooling
    x = layers.GlobalAveragePooling2D(name='global_avg_pool')(x)
    
    # Dense layer with dropout for regularization
    x = layers.Dense(config.DENSE_UNITS, activation='relu', name='dense_512')(x)
    x = layers.Dropout(config.DROPOUT_RATE, name='dropout')(x)
    
    # Output layer
    outputs = layers.Dense(num_classes, activation='softmax', name='output')(x)
    
    # Create model
    model = models.Model(inputs=inputs, outputs=outputs, name='plant_health_resnet50')
    
    # Compile model
    model.compile(
        optimizer=Adam(learning_rate=config.LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy', keras.metrics.TopKCategoricalAccuracy(k=5, name='top_5_accuracy')]
    )
    
    print(f"\n📊 Model Summary:")
    print(f"   Total Parameters: {model.count_params():,}")
    print(f"   Trainable Parameters: {sum([tf.size(w).numpy() for w in model.trainable_weights]):,}")
    print(f"   Non-trainable Parameters: {sum([tf.size(w).numpy() for w in model.non_trainable_weights]):,}")
    print(f"   Output Classes: {num_classes}")
    print(f"   Optimizer: Adam (lr={config.LEARNING_RATE})")
    print(f"   Loss: Categorical Crossentropy")
    print(f"{'='*60}\n")
    
    return model


def unfreeze_base_model(model, num_layers_to_unfreeze=20):
    """
    Unfreeze top layers of base model for fine-tuning
    
    Args:
        model: Compiled Keras model
        num_layers_to_unfreeze: Number of top layers to unfreeze
        
    Returns:
        keras.Model: Model with unfrozen layers, recompiled
    """
    # Get the base model (ResNet50)
    base_model = model.layers[1]  # Assuming base model is second layer
    
    # Unfreeze the top layers
    base_model.trainable = True
    
    # Freeze all layers except the top ones
    for layer in base_model.layers[:-num_layers_to_unfreeze]:
        layer.trainable = False
    
    # Recompile with lower learning rate for fine-tuning
    model.compile(
        optimizer=Adam(learning_rate=config.LEARNING_RATE / 10),
        loss='categorical_crossentropy',
        metrics=['accuracy', keras.metrics.TopKCategoricalAccuracy(k=5, name='top_5_accuracy')]
    )
    
    print(f"\n🔓 Unfroze top {num_layers_to_unfreeze} layers for fine-tuning")
    print(f"   New learning rate: {config.LEARNING_RATE / 10}")
    
    return model


def estimate_severity(prediction_confidence):
    """
    Estimate severity based on model prediction confidence
    
    Args:
        prediction_confidence: Confidence score (0-1) from model prediction
        
    Returns:
        str: Severity level (Mild, Moderate, Severe/Uncertain)
    """
    if prediction_confidence >= config.SEVERITY_THRESHOLDS['high_confidence']:
        return "Mild/Minor"
    elif prediction_confidence >= config.SEVERITY_THRESHOLDS['medium_confidence']:
        return "Moderate"
    else:
        return "Severe/Uncertain"


def get_severity_color(severity):
    """
    Get color code for severity level (for terminal output)
    
    Args:
        severity: Severity string
        
    Returns:
        str: ANSI color code
    """
    if severity == "Mild/Minor":
        return "\033[92m"  # Green
    elif severity == "Moderate":
        return "\033[93m"  # Yellow
    else:
        return "\033[91m"  # Red


def predict_with_severity(model, image_array, class_names):
    """
    Make prediction and estimate severity
    
    Args:
        model: Trained Keras model
        image_array: Preprocessed image array
        class_names: List of class names
        
    Returns:
        dict: Prediction results with severity estimation
    """
    # Make prediction
    predictions = model.predict(image_array, verbose=0)
    
    # Get top prediction
    top_idx = np.argmax(predictions[0])
    top_confidence = predictions[0][top_idx]
    top_class = class_names[top_idx] if class_names else f"Class_{top_idx}"
    
    # Get top 5 predictions
    top_5_idx = np.argsort(predictions[0])[-5:][::-1]
    top_5_predictions = [
        {
            'class': class_names[idx] if class_names else f"Class_{idx}",
            'confidence': float(predictions[0][idx])
        }
        for idx in top_5_idx
    ]
    
    # Estimate severity
    severity = estimate_severity(top_confidence)
    
    return {
        'predicted_class': top_class,
        'confidence': float(top_confidence),
        'severity': severity,
        'top_5_predictions': top_5_predictions
    }


def display_model_architecture(model):
    """
    Display detailed model architecture
    
    Args:
        model: Keras model
    """
    print("\n" + "="*80)
    print("MODEL ARCHITECTURE")
    print("="*80)
    model.summary()
    print("="*80 + "\n")


if __name__ == '__main__':
    # Test model building
    print("Testing model architecture...")
    test_model = build_model(num_classes=50, freeze_base=True)
    display_model_architecture(test_model)
    
    # Test severity estimation
    test_confidences = [0.95, 0.75, 0.45, 0.20]
    print("\n🧪 Testing Severity Estimation:")
    for conf in test_confidences:
        sev = estimate_severity(conf)
        color = get_severity_color(sev)
        print(f"   Confidence: {conf:.2f} → {color}{sev}\033[0m")
    
    print("\n✅ Model module tested successfully")
