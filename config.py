"""
Configuration file for Plant Health Detection System
Contains all hyperparameters, paths, and settings
"""

import os

# ==================== PATHS ====================
# Base directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.dirname(BASE_DIR)

# Dataset paths
PLANTS_BASE_PATH = os.path.join(DATA_DIR, "plants", "New Plant Diseases Dataset(Augmented)")
PLANTS_TRAIN_PATH = os.path.join(PLANTS_BASE_PATH, "train")
PLANTS_VALID_PATH = os.path.join(PLANTS_BASE_PATH, "valid")

PESTS_PATH = os.path.join(DATA_DIR, "pests")

# Model paths
MODELS_DIR = os.path.join(BASE_DIR, "models")
MODEL_SAVE_PATH = os.path.join(MODELS_DIR, "plant_health_model.h5")
CLASS_MAPPING_PATH = os.path.join(MODELS_DIR, "class_mapping.json")
TRAINING_HISTORY_PATH = os.path.join(MODELS_DIR, "training_history.csv")

# ==================== MODEL HYPERPARAMETERS ====================
# Image settings
IMG_HEIGHT = 224
IMG_WIDTH = 224
IMG_CHANNELS = 3
IMG_SIZE = (IMG_HEIGHT, IMG_WIDTH)

# Training settings
BATCH_SIZE = 32
EPOCHS = 30
LEARNING_RATE = 0.0001

# Pest dataset split (since it doesn't have predefined splits)
PEST_TRAIN_SPLIT = 0.8  # 80% training, 20% validation

# ==================== DATA AUGMENTATION ====================
# Training augmentation parameters
AUGMENTATION_CONFIG = {
    'rotation_range': 20,
    'width_shift_range': 0.1,
    'height_shift_range': 0.1,
    'brightness_range': [0.8, 1.2],
    'zoom_range': 0.2,
    'horizontal_flip': True,
    'fill_mode': 'nearest'
}

# ==================== MODEL ARCHITECTURE ====================
# ResNet50 settings
PRETRAINED_MODEL = 'ResNet50'
INCLUDE_TOP = False
WEIGHTS = 'imagenet'

# Custom head settings
DENSE_UNITS = 512
DROPOUT_RATE = 0.5

# ==================== TRAINING CALLBACKS ====================
# Early stopping
EARLY_STOPPING_PATIENCE = 10
EARLY_STOPPING_MONITOR = 'val_loss'

# Learning rate reduction
REDUCE_LR_PATIENCE = 5
REDUCE_LR_FACTOR = 0.5
REDUCE_LR_MIN_LR = 1e-7

# Model checkpoint
CHECKPOINT_MONITOR = 'val_accuracy'
CHECKPOINT_MODE = 'max'

# ==================== SEVERITY ESTIMATION ====================
# Confidence thresholds for severity estimation
SEVERITY_THRESHOLDS = {
    'high_confidence': 0.8,    # > 0.8: Mild/Minor
    'medium_confidence': 0.5,  # 0.5-0.8: Moderate
    # < 0.5: Severe or Uncertain
}

# ==================== OTHER SETTINGS ====================
# Random seed for reproducibility
RANDOM_SEED = 42

# Validation settings
MIN_IMAGES_PER_CLASS = 10  # Minimum images required per class

# Normalization mode
NORMALIZATION = 'rescale'  # Options: 'rescale' (0-1) or 'imagenet' (ImageNet stats)

# Class name prefix for combined dataset
PLANT_CLASS_PREFIX = "plant"
PEST_CLASS_PREFIX = "pest"
