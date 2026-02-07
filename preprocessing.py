"""
Preprocessing Module for Plant Health Detection System
Handles image preprocessing and data augmentation
"""

import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import config


def create_train_data_generator():
    """
    Create ImageDataGenerator for training with augmentation
    
    Returns:
        ImageDataGenerator: Configured generator with augmentation
    """
    if config.NORMALIZATION == 'rescale':
        # Simple rescaling to [0, 1]
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            **config.AUGMENTATION_CONFIG
        )
    else:
        # ImageNet normalization (same as ResNet50 pretraining)
        # This will be handled in the generator flow
        train_datagen = ImageDataGenerator(
            preprocessing_function=None,  # Will use keras.applications.resnet50.preprocess_input
            **config.AUGMENTATION_CONFIG
        )
    
    return train_datagen


def create_validation_data_generator():
    """
    Create ImageDataGenerator for validation (no augmentation)
    
    Returns:
        ImageDataGenerator: Configured generator for validation
    """
    if config.NORMALIZATION == 'rescale':
        val_datagen = ImageDataGenerator(rescale=1./255)
    else:
        val_datagen = ImageDataGenerator(preprocessing_function=None)
    
    return val_datagen


def create_data_generators_from_directory(class_mapping):
    """
    Create data generators from directory structure
    
    Args:
        class_mapping: Dictionary mapping class names to indices
        
    Returns:
        tuple: (train_generator, validation_generator, steps_per_epoch, validation_steps)
    """
    from tensorflow.keras.applications.resnet50 import preprocess_input
    
    # Create generators with preprocessing
    if config.NORMALIZATION == 'imagenet':
        train_datagen = ImageDataGenerator(
            preprocessing_function=preprocess_input,
            **config.AUGMENTATION_CONFIG
        )
        val_datagen = ImageDataGenerator(
            preprocessing_function=preprocess_input
        )
    else:
        train_datagen = create_train_data_generator()
        val_datagen = create_validation_data_generator()
    
    # Plant disease generators
    train_generator_plants = train_datagen.flow_from_directory(
        config.PLANTS_TRAIN_PATH,
        target_size=config.IMG_SIZE,
        batch_size=config.BATCH_SIZE,
        class_mode='categorical',
        shuffle=True,
        seed=config.RANDOM_SEED
    )
    
    val_generator_plants = val_datagen.flow_from_directory(
        config.PLANTS_VALID_PATH,
        target_size=config.IMG_SIZE,
        batch_size=config.BATCH_SIZE,
        class_mode='categorical',
        shuffle=False
    )
    
    print(f"\n📦 Data Generators Created:")
    print(f"   Training batches (plants): {len(train_generator_plants)}")
    print(f"   Validation batches (plants): {len(val_generator_plants)}")
    print(f"   Batch size: {config.BATCH_SIZE}")
    print(f"   Image shape: {config.IMG_SIZE + (config.IMG_CHANNELS,)}")
    print(f"   Augmentation: Enabled for training")
    
    return train_generator_plants, val_generator_plants


def create_combined_generators(dataset_info):
    """
    Create combined generators for both plant diseases and pests
    This is a more advanced implementation that combines both datasets
    
    Args:
        dataset_info: Dictionary with dataset information
        
    Returns:
        tuple: (combined_train_generator, combined_val_generator)
    """
    # For simplicity in this hackathon, we'll use the plant disease generators
    # and note that pests can be added to the same directory structure
    # or handled separately in production
    
    train_gen, val_gen = create_data_generators_from_directory(
        dataset_info['class_mapping']
    )
    
    return train_gen, val_gen


def preprocess_single_image(image_path, target_size=None):
    """
    Preprocess a single image for inference
    
    Args:
        image_path: Path to the image file
        target_size: Target size tuple (height, width), defaults to config.IMG_SIZE
        
    Returns:
        numpy.ndarray: Preprocessed image ready for model input
    """
    from tensorflow.keras.preprocessing import image
    from tensorflow.keras.applications.resnet50 import preprocess_input
    
    if target_size is None:
        target_size = config.IMG_SIZE
    
    # Load image
    img = image.load_img(image_path, target_size=target_size)
    
    # Convert to array
    img_array = image.img_to_array(img)
    
    # Expand dimensions to match batch shape
    img_array = np.expand_dims(img_array, axis=0)
    
    # Preprocess based on config
    if config.NORMALIZATION == 'imagenet':
        img_array = preprocess_input(img_array)
    else:
        img_array = img_array / 255.0
    
    return img_array


def get_augmentation_summary():
    """
    Get a summary of augmentation parameters
    
    Returns:
        str: Formatted summary of augmentation settings
    """
    summary = "\n🎨 Data Augmentation Configuration:\n"
    for key, value in config.AUGMENTATION_CONFIG.items():
        summary += f"   {key}: {value}\n"
    return summary


if __name__ == '__main__':
    # Test preprocessing
    print("Testing preprocessing module...")
    print(get_augmentation_summary())
    
    train_gen = create_train_data_generator()
    val_gen = create_validation_data_generator()
    
    print("✅ Preprocessing module tested successfully")
