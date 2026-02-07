"""
Main Training Script for Plant Health Detection System
Orchestrates the complete training pipeline
"""

import os
import sys
import argparse
import numpy as np
import tensorflow as tf
from tensorflow import keras
from datetime import datetime

# Import custom modules
import config
from data_loader import load_and_validate_datasets
from preprocessing import create_data_generators_from_directory, get_augmentation_summary
from model import build_model, display_model_architecture
from utils import save_model, plot_training_history


def set_random_seeds(seed=42):
    """Set random seeds for reproducibility"""
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def create_callbacks():
    """
    Create training callbacks
    
    Returns:
        list: List of Keras callbacks
    """
    callbacks = []
    
    # Model checkpoint - save best model
    checkpoint_callback = keras.callbacks.ModelCheckpoint(
        filepath=config.MODEL_SAVE_PATH,
        monitor=config.CHECKPOINT_MONITOR,
        mode=config.CHECKPOINT_MODE,
        save_best_only=True,
        verbose=1
    )
    callbacks.append(checkpoint_callback)
    
    # Early stopping
    early_stopping_callback = keras.callbacks.EarlyStopping(
        monitor=config.EARLY_STOPPING_MONITOR,
        patience=config.EARLY_STOPPING_PATIENCE,
        restore_best_weights=True,
        verbose=1
    )
    callbacks.append(early_stopping_callback)
    
    # Reduce learning rate on plateau
    reduce_lr_callback = keras.callbacks.ReduceLROnPlateau(
        monitor=config.EARLY_STOPPING_MONITOR,
        factor=config.REDUCE_LR_FACTOR,
        patience=config.REDUCE_LR_PATIENCE,
        min_lr=config.REDUCE_LR_MIN_LR,
        verbose=1
    )
    callbacks.append(reduce_lr_callback)
    
    # CSV Logger
    csv_logger_callback = keras.callbacks.CSVLogger(
        filename=config.TRAINING_HISTORY_PATH,
        separator=',',
        append=False
    )
    callbacks.append(csv_logger_callback)
    
    # TensorBoard (optional)
    tensorboard_dir = os.path.join(config.MODELS_DIR, 'tensorboard_logs')
    os.makedirs(tensorboard_dir, exist_ok=True)
    tensorboard_callback = keras.callbacks.TensorBoard(
        log_dir=tensorboard_dir,
        histogram_freq=0,
        write_graph=True
    )
    callbacks.append(tensorboard_callback)
    
    print(f"\n✅ Callbacks configured:")
    print(f"   - ModelCheckpoint: Save best model based on {config.CHECKPOINT_MONITOR}")
    print(f"   - EarlyStopping: Patience = {config.EARLY_STOPPING_PATIENCE}")
    print(f"   - ReduceLROnPlateau: Patience = {config.REDUCE_LR_PATIENCE}")
    print(f"   - CSVLogger: Log to {config.TRAINING_HISTORY_PATH}")
    print(f"   - TensorBoard: Log to {tensorboard_dir}")
    
    return callbacks


def train_model(epochs=None, batch_size=None):
    """
    Main training function
    
    Args:
        epochs: Number of training epochs (optional, uses config if None)
        batch_size: Batch size (optional, uses config if None)
    """
    # Override config if provided
    if epochs:
        config.EPOCHS = epochs
    if batch_size:
        config.BATCH_SIZE = batch_size
    
    # Set random seeds
    set_random_seeds(config.RANDOM_SEED)
    
    print("\n" + "="*80)
    print("🌱 PLANT HEALTH DETECTION SYSTEM - TRAINING PIPELINE")
    print("="*80)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"TensorFlow version: {tf.__version__}")
    print(f"GPU Available: {len(tf.config.list_physical_devices('GPU')) > 0}")
    print("="*80)
    
    # Step 1: Load and validate datasets
    print("\n📂 STEP 1: Loading and Validating Datasets")
    print("-" * 80)
    dataset_info = load_and_validate_datasets()
    
    # Step 2: Create data generators
    print("\n🔄 STEP 2: Creating Data Generators")
    print("-" * 80)
    print(get_augmentation_summary())
    
    train_generator, val_generator = create_data_generators_from_directory(
        dataset_info['class_mapping']
    )
    
    num_classes = len(train_generator.class_indices)
    print(f"\n✅ Data generators ready with {num_classes} classes")
    
    # Step 3: Build model
    print("\n🏗️  STEP 3: Building Model")
    print("-" * 80)
    model = build_model(num_classes=num_classes, freeze_base=True)
    
    # Display architecture (summary)
    if '--verbose' in sys.argv or '-v' in sys.argv:
        display_model_architecture(model)
    
    # Step 4: Setup callbacks
    print("\n⚙️  STEP 4: Configuring Training Callbacks")
    print("-" * 80)
    callbacks = create_callbacks()
    
    # Step 5: Train model
    print("\n🚀 STEP 5: Training Model")
    print("-" * 80)
    print(f"Training for {config.EPOCHS} epochs with batch size {config.BATCH_SIZE}")
    print(f"Training samples: {train_generator.samples}")
    print(f"Validation samples: {val_generator.samples}")
    print(f"Steps per epoch: {len(train_generator)}")
    print(f"Validation steps: {len(val_generator)}")
    print("-" * 80)
    
    history = model.fit(
        train_generator,
        epochs=config.EPOCHS,
        validation_data=val_generator,
        callbacks=callbacks,
        verbose=1
    )
    
    # Step 6: Evaluate model
    print("\n📊 STEP 6: Evaluating Model")
    print("-" * 80)
    
    final_train_acc = history.history['accuracy'][-1]
    final_val_acc = history.history['val_accuracy'][-1]
    final_train_loss = history.history['loss'][-1]
    final_val_loss = history.history['val_loss'][-1]
    
    print(f"\n🎯 Final Training Results:")
    print(f"   Training Accuracy: {final_train_acc:.4f} ({final_train_acc*100:.2f}%)")
    print(f"   Training Loss: {final_train_loss:.4f}")
    print(f"   Validation Accuracy: {final_val_acc:.4f} ({final_val_acc*100:.2f}%)")
    print(f"   Validation Loss: {final_val_loss:.4f}")
    
    # Check for overfitting
    if (final_train_acc - final_val_acc) > 0.1:
        print(f"\n⚠️  Warning: Possible overfitting detected (train-val gap: {(final_train_acc - final_val_acc)*100:.2f}%)")
    
    # Step 7: Save results
    print("\n💾 STEP 7: Saving Results")
    print("-" * 80)
    
    # Save model
    save_model(model)
    
    # Plot training history
    plot_history_path = os.path.join(config.MODELS_DIR, 'training_history.png')
    plot_training_history(history, save_path=plot_history_path)
    
    # Save class indices
    import json
    class_indices_path = os.path.join(config.MODELS_DIR, 'class_indices.json')
    with open(class_indices_path, 'w') as f:
        json.dump(train_generator.class_indices, f, indent=2)
    print(f"✅ Class indices saved to: {class_indices_path}")
    
    # Step 8: Demonstration of severity estimation
    print("\n🧪 STEP 8: Severity Estimation Demonstration")
    print("-" * 80)
    from model import estimate_severity, get_severity_color
    
    print("\nSeverity Estimation based on Prediction Confidence:")
    test_confidences = [0.95, 0.85, 0.75, 0.65, 0.50, 0.35, 0.20]
    for conf in test_confidences:
        severity = estimate_severity(conf)
        color = get_severity_color(severity)
        print(f"   Confidence: {conf:.2f} → {color}{severity}\033[0m")
    
    # Final summary
    print("\n" + "="*80)
    print("✅ TRAINING COMPLETED SUCCESSFULLY")
    print("="*80)
    print(f"Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\n📁 Generated Files:")
    print(f"   - Model: {config.MODEL_SAVE_PATH}")
    print(f"   - Class Mapping: {config.CLASS_MAPPING_PATH}")
    print(f"   - Class Indices: {class_indices_path}")
    print(f"   - Training History: {config.TRAINING_HISTORY_PATH}")
    print(f"   - Training Plot: {plot_history_path}")
    print(f"\n🎯 Best Validation Accuracy: {max(history.history['val_accuracy'])*100:.2f}%")
    print("="*80)
    
    return model, history


if __name__ == '__main__':
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train Plant Health Detection Model')
    parser.add_argument('--epochs', type=int, default=None, 
                       help=f'Number of training epochs (default: {config.EPOCHS})')
    parser.add_argument('--batch-size', type=int, default=None,
                       help=f'Batch size (default: {config.BATCH_SIZE})')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Show detailed model architecture')
    
    args = parser.parse_args()
    
    try:
        # Train the model
        model, history = train_model(epochs=args.epochs, batch_size=args.batch_size)
        
        print("\n✨ Training completed! Your model is ready for deployment.")
        print("\n📖 Next Steps:")
        print("   1. Test the model with new images using utils.predict_single_image()")
        print("   2. Integrate the model into a Flask or FastAPI backend")
        print("   3. Use the severity estimation for actionable insights")
        
    except Exception as e:
        print(f"\n❌ Error during training: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
