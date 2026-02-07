"""
Utility Functions for Plant Health Detection System
Provides helper functions for model saving, visualization, and inference
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow import keras
from sklearn.metrics import classification_report, confusion_matrix
import config


def save_model(model, model_path=None):
    """
    Save trained model in Keras format
    
    Args:
        model: Trained Keras model
        model_path: Path to save model (optional)
    """
    if model_path is None:
        model_path = config.MODEL_SAVE_PATH
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    
    # Save in .keras format (TensorFlow 2.x recommended)
    model.save(model_path)
    print(f"✅ Model saved to: {model_path}")
    
    # Also save in .h5 format for compatibility
    h5_path = model_path.replace('.keras', '.h5')
    model.save(h5_path)
    print(f"✅ Model also saved to: {h5_path}")


def load_model(model_path=None):
    """
    Load trained model
    
    Args:
        model_path: Path to saved model (optional)
        
    Returns:
        keras.Model: Loaded model
    """
    if model_path is None:
        model_path = config.MODEL_SAVE_PATH
    
    model = keras.models.load_model(model_path)
    print(f"✅ Model loaded from: {model_path}")
    return model


def load_class_mapping(mapping_path=None):
    """
    Load class mapping from JSON file
    
    Args:
        mapping_path: Path to class mapping file (optional)
        
    Returns:
        dict: Class name to index mapping
    """
    if mapping_path is None:
        mapping_path = config.CLASS_MAPPING_PATH
    
    with open(mapping_path, 'r') as f:
        class_mapping = json.load(f)
    
    return class_mapping


def plot_training_history(history, save_path=None):
    """
    Plot training history (accuracy and loss curves)
    
    Args:
        history: Keras History object or dict with history
        save_path: Path to save plot (optional)
    """
    if hasattr(history, 'history'):
        history = history.history
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot accuracy
    axes[0].plot(history['accuracy'], label='Training Accuracy', linewidth=2)
    axes[0].plot(history['val_accuracy'], label='Validation Accuracy', linewidth=2)
    axes[0].set_title('Model Accuracy Over Epochs', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Accuracy', fontsize=12)
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    
    # Plot loss
    axes[1].plot(history['loss'], label='Training Loss', linewidth=2)
    axes[1].plot(history['val_loss'], label='Validation Loss', linewidth=2)
    axes[1].set_title('Model Loss Over Epochs', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Loss', fontsize=12)
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Training history plot saved to: {save_path}")
    else:
        save_path = os.path.join(config.MODELS_DIR, 'training_history.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Training history plot saved to: {save_path}")
    
    plt.close()


def generate_confusion_matrix(model, data_generator, class_names, save_path=None):
    """
    Generate and plot confusion matrix
    
    Args:
        model: Trained Keras model
        data_generator: Data generator for predictions
        class_names: List of class names
        save_path: Path to save plot (optional)
    """
    # Get predictions
    y_pred = model.predict(data_generator, verbose=1)
    y_pred_classes = np.argmax(y_pred, axis=1)
    
    # Get true labels
    y_true = data_generator.classes
    
    # Generate confusion matrix
    cm = confusion_matrix(y_true, y_pred_classes)
    
    # Plot (only if classes are not too many)
    if len(class_names) <= 20:
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names)
        plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            save_path = os.path.join(config.MODELS_DIR, 'confusion_matrix.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        print(f"✅ Confusion matrix saved to: {save_path}")
        plt.close()
    else:
        print(f"⚠️  Too many classes ({len(class_names)}) to plot confusion matrix")
    
    return cm


def generate_classification_report(model, data_generator, class_names, save_path=None):
    """
    Generate classification report
    
    Args:
        model: Trained Keras model
        data_generator: Data generator for predictions
        class_names: List of class names
        save_path: Path to save report (optional)
        
    Returns:
        str: Classification report
    """
    # Get predictions
    y_pred = model.predict(data_generator, verbose=1)
    y_pred_classes = np.argmax(y_pred, axis=1)
    
    # Get true labels
    y_true = data_generator.classes
    
    # Generate report
    report = classification_report(y_true, y_pred_classes, 
                                   target_names=class_names,
                                   digits=3)
    
    print("\n" + "="*80)
    print("CLASSIFICATION REPORT")
    print("="*80)
    print(report)
    print("="*80 + "\n")
    
    # Save to file
    if save_path is None:
        save_path = os.path.join(config.MODELS_DIR, 'classification_report.txt')
    
    with open(save_path, 'w') as f:
        f.write(report)
    
    print(f"✅ Classification report saved to: {save_path}")
    
    return report


def predict_single_image(model, image_path, class_mapping):
    """
    Predict class for a single image with severity estimation
    
    Args:
        model: Trained Keras model
        image_path: Path to image file
        class_mapping: Dictionary mapping class names to indices
        
    Returns:
        dict: Prediction results with severity
    """
    from preprocessing import preprocess_single_image
    from model import predict_with_severity
    
    # Preprocess image
    img_array = preprocess_single_image(image_path)
    
    # Get class names list
    class_names = [k for k, v in sorted(class_mapping.items(), key=lambda x: x[1])]
    
    # Make prediction with severity
    results = predict_with_severity(model, img_array, class_names)
    
    return results


def display_prediction_results(image_path, results):
    """
    Display prediction results in formatted output
    
    Args:
        image_path: Path to predicted image
        results: Prediction results dictionary
    """
    from model import get_severity_color
    
    print("\n" + "="*80)
    print("🔍 PREDICTION RESULTS")
    print("="*80)
    print(f"Image: {os.path.basename(image_path)}")
    print(f"\n🎯 Predicted Class: {results['predicted_class']}")
    print(f"📊 Confidence: {results['confidence']:.2%}")
    
    severity_color = get_severity_color(results['severity'])
    print(f"⚕️  Severity: {severity_color}{results['severity']}\033[0m")
    
    print(f"\n📈 Top 5 Predictions:")
    for i, pred in enumerate(results['top_5_predictions'], 1):
        print(f"   {i}. {pred['class']}: {pred['confidence']:.2%}")
    
    print("="*80 + "\n")


def create_inference_function(model_path=None, class_mapping_path=None):
    """
    Create a standalone inference function for deployment
    
    Args:
        model_path: Path to saved model
        class_mapping_path: Path to class mapping file
        
    Returns:
        function: Inference function
    """
    # Load model and mapping
    model = load_model(model_path)
    class_mapping = load_class_mapping(class_mapping_path)
    
    def inference(image_path):
        """
        Perform inference on a single image
        
        Args:
            image_path: Path to image file
            
        Returns:
            dict: Prediction results
        """
        return predict_single_image(model, image_path, class_mapping)
    
    return inference


if __name__ == '__main__':
    print("Testing utility functions...")
    print("✅ Utilities module ready")
