"""
Data Loader Module for Plant Health Detection System
Handles dataset loading, validation, and preprocessing
"""

import os
import json
import numpy as np
from pathlib import Path
from collections import Counter
import config


def validate_dataset_structure(base_path, dataset_name="Dataset"):
    """
    Validate dataset directory structure and count images per class
    
    Args:
        base_path: Path to dataset directory
        dataset_name: Name of the dataset for logging
        
    Returns:
        dict: Dataset information including classes, counts, and validation status
    """
    print(f"\n{'='*60}")
    print(f"Validating {dataset_name}")
    print(f"{'='*60}")
    
    if not os.path.exists(base_path):
        raise FileNotFoundError(f"Dataset path not found: {base_path}")
    
    # Get all subdirectories (classes)
    classes = sorted([d for d in os.listdir(base_path) 
                     if os.path.isdir(os.path.join(base_path, d)) and not d.startswith('.')])
    
    if not classes:
        raise ValueError(f"No class directories found in {base_path}")
    
    # Count images per class
    class_counts = {}
    total_images = 0
    corrupted_files = []
    
    for class_name in classes:
        class_path = os.path.join(base_path, class_name)
        
        # Count valid image files
        image_files = [f for f in os.listdir(class_path) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        
        num_images = len(image_files)
        class_counts[class_name] = num_images
        total_images += num_images
        
        # Warn if too few images
        if num_images < config.MIN_IMAGES_PER_CLASS:
            print(f"⚠️  Warning: Class '{class_name}' has only {num_images} images")
    
    # Display statistics
    print(f"\n📊 Dataset Statistics:")
    print(f"   Total Classes: {len(classes)}")
    print(f"   Total Images: {total_images:,}")
    print(f"   Average Images/Class: {total_images/len(classes):.1f}")
    print(f"   Min Images/Class: {min(class_counts.values())}")
    print(f"   Max Images/Class: {max(class_counts.values())}")
    
    # Show some class examples
    print(f"\n📁 Sample Classes:")
    for i, (class_name, count) in enumerate(list(class_counts.items())[:5]):
        print(f"   {class_name}: {count} images")
    if len(classes) > 5:
        print(f"   ... and {len(classes) - 5} more classes")
    
    return {
        'classes': classes,
        'class_counts': class_counts,
        'total_images': total_images,
        'num_classes': len(classes),
        'valid': True
    }


def split_pest_dataset(pests_path, train_split=0.8, seed=42):
    """
    Split pest dataset into train and validation sets
    
    Args:
        pests_path: Path to pests directory
        train_split: Fraction for training (0.0 - 1.0)
        seed: Random seed for reproducibility
        
    Returns:
        tuple: (train_data_list, val_data_list) where each is list of (file_path, class_idx)
    """
    np.random.seed(seed)
    
    train_data = []
    val_data = []
    
    pest_classes = sorted([d for d in os.listdir(pests_path)
                          if os.path.isdir(os.path.join(pests_path, d)) and not d.startswith('.')])
    
    for class_idx, class_name in enumerate(pest_classes):
        class_path = os.path.join(pests_path, class_name)
        
        # Get all image files
        image_files = [os.path.join(class_path, f) for f in os.listdir(class_path)
                      if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        
        # Shuffle and split
        np.random.shuffle(image_files)
        split_idx = int(len(image_files) * train_split)
        
        train_files = image_files[:split_idx]
        val_files = image_files[split_idx:]
        
        train_data.extend([(f, class_name) for f in train_files])
        val_data.extend([(f, class_name) for f in val_files])
    
    return train_data, val_data


def create_unified_class_mapping():
    """
    Create unified class mapping for both plants and pests
    
    Returns:
        dict: Mapping of class names to indices
    """
    all_classes = []
    
    # Get plant disease classes
    if os.path.exists(config.PLANTS_TRAIN_PATH):
        plant_classes = sorted([d for d in os.listdir(config.PLANTS_TRAIN_PATH)
                               if os.path.isdir(os.path.join(config.PLANTS_TRAIN_PATH, d)) 
                               and not d.startswith('.')])
        all_classes.extend(plant_classes)
    
    # Get pest classes
    if os.path.exists(config.PESTS_PATH):
        pest_classes = sorted([d for d in os.listdir(config.PESTS_PATH)
                              if os.path.isdir(os.path.join(config.PESTS_PATH, d))
                              and not d.startswith('.')])
        # Prefix pest classes to distinguish them
        pest_classes = [f"pest_{pc}" for pc in pest_classes]
        all_classes.extend(pest_classes)
    
    # Create mapping
    class_mapping = {class_name: idx for idx, class_name in enumerate(sorted(all_classes))}
    
    return class_mapping, all_classes


def load_and_validate_datasets():
    """
    Load and validate both plant disease and pest datasets
    
    Returns:
        dict: Complete dataset information including class mappings and statistics
    """
    print("\n" + "="*80)
    print("🌱 PLANT HEALTH DETECTION SYSTEM - Dataset Validation")
    print("="*80)
    
    # Validate plant disease training set
    plants_train_info = validate_dataset_structure(
        config.PLANTS_TRAIN_PATH, 
        "Plant Disease Training Set"
    )
    
    # Validate plant disease validation set
    plants_val_info = validate_dataset_structure(
        config.PLANTS_VALID_PATH,
        "Plant Disease Validation Set"
    )
    
    # Validate pest dataset
    pests_info = validate_dataset_structure(
        config.PESTS_PATH,
        "Pest Dataset (will be split)"
    )
    
    # Create unified class mapping
    class_mapping, all_class_names = create_unified_class_mapping()
    
    # Summary
    total_classes = len(class_mapping)
    total_train_images = plants_train_info['total_images']
    total_val_images = plants_val_info['total_images']
    total_pest_images = pests_info['total_images']
    
    # Calculate pest split
    pest_train_count = int(total_pest_images * config.PEST_TRAIN_SPLIT)
    pest_val_count = total_pest_images - pest_train_count
    
    print(f"\n{'='*80}")
    print("📈 UNIFIED DATASET SUMMARY")
    print(f"{'='*80}")
    print(f"Total Classes: {total_classes}")
    print(f"  - Plant Disease Classes: {plants_train_info['num_classes']}")
    print(f"  - Pest Classes: {pests_info['num_classes']}")
    print(f"\nTotal Training Images: {total_train_images + pest_train_count:,}")
    print(f"  - Plant Diseases: {total_train_images:,}")
    print(f"  - Pests: {pest_train_count:,}")
    print(f"\nTotal Validation Images: {total_val_images + pest_val_count:,}")
    print(f"  - Plant Diseases: {total_val_images:,}")
    print(f"  - Pests: {pest_val_count:,}")
    print(f"{'='*80}\n")
    
    # Save class mapping
    os.makedirs(config.MODELS_DIR, exist_ok=True)
    with open(config.CLASS_MAPPING_PATH, 'w') as f:
        json.dump(class_mapping, f, indent=2)
    print(f"✅ Class mapping saved to: {config.CLASS_MAPPING_PATH}\n")
    
    return {
        'class_mapping': class_mapping,
        'all_classes': all_class_names,
        'num_classes': total_classes,
        'plants_train': plants_train_info,
        'plants_val': plants_val_info,
        'pests': pests_info,
        'total_train_images': total_train_images + pest_train_count,
        'total_val_images': total_val_images + pest_val_count
    }


if __name__ == '__main__':
    # Test the data loader
    dataset_info = load_and_validate_datasets()
    print(f"\n✅ Dataset validation completed successfully!")
    print(f"Ready to train with {dataset_info['num_classes']} classes")
